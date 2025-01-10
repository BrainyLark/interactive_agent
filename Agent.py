import os
import logging
import requests
import operator

from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.utils import convert_to_secret_str
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage, HumanMessage

class Decision(Enum):
    THINK = 0
    ACT = 1
    END = 2

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    
class Agent:
    
    def __init__(self, model: ChatOpenAI, tools: List[BaseTool], system:str=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.act)
        graph.add_conditional_edges("llm", self.next_move, {Decision.THINK: "llm", Decision.ACT: "action", Decision.END: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = { t.name : t for t in tools }
        self.model = model.bind_tools(tools)
        
    def call_model(self, state: AgentState) -> dict:
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            
        messages = self.model.invoke(messages)
        return { "messages" : [messages] }
    
    def next_move(self, state: AgentState) -> Literal[Decision.THINK, Decision.ACT, Decision.END]:
        llm_message = state["messages"][-1]
        if type(llm_message) is AIMessage:
            if len(llm_message.content) > 0 and len(llm_message.tool_calls) == 0:
                if (type(llm_message.content) is str) and ("final state:" in llm_message.content.lower()):
                     return Decision.END
                print(f"{llm_message.content}")
                return Decision.THINK
            elif len(llm_message.tool_calls) > 0 and len(llm_message.content) == 0:
                print(f"ACTION: {llm_message.tool_calls[0]['name']}({llm_message.tool_calls[0]['args']})")
                return Decision.ACT
            elif len(llm_message.tool_calls) > 0 and len(llm_message.content) > 0:
                print("CONTENT and TOOL_CALLS both satisfied!")
                print(f"ACTION: {llm_message.tool_calls[0]['name']}({llm_message.tool_calls[0]['args']})")
                return Decision.ACT
                
        return Decision.END
    
    def act(self, state: AgentState) -> dict | None:
        llm_message = state['messages'][-1]
        if not type(llm_message) is AIMessage:
            return
        results = []
        for t in llm_message.tool_calls:
            if not t['name'] in self.tools:
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            
        return { "messages" : results }
    
    
@tool
def set_appointment(appointment_time: str, branch: int, name: str, expected_duration: int) -> str:
    """Call this function to insert appointment instance into the database table. Make sure the data validation and integrity of
    the appointment argument is upheld. If there is a lack of property values, please use `take_user_input` tool to elicit the
    necessary information from the user.

    Args:
        appointment_time - datetime in ISO format as to when user set the appointment to
        branch - what branch the customer got his appointment
        name - the name of the user who reserved the slot
        expected_duration - the amount of time in minutes how long the hair cut would take

    Returns:
        str: string representation of whether the insertion was successful.
    """
    
    appointment = { "id": None, 
                   "name": name, 
                   "operation": None, 
                   "branch": branch,
                   "expected_duration": expected_duration, 
                   "created_date": datetime.now().isoformat(),
                   "appointment_datetime": appointment_time
                }
    
    response = requests.post("http://localhost:8000/appointment", json=appointment)
    response_parsed = response.json()
    
    print(f"Таны захиалгыг бүртгэлээ. {response_parsed}")
    
    return response_parsed
    

@tool
def check_conflicting_appointment(dt: str, duration: int) -> str:
    """Call this function to check if there is a conflicting appointment at barbershop with the requested datetime `dt` that will take `duration` minutes.

    Args:
        dt: str - appointment starting datetime in string ISO format
        duration: int - expected amount of time barber might take in minutes

    Returns:
        str: tells if the requested datetime slot is free to make an appointment, otherwise, it will suggest another time when it is free.
    """
    
    print(f"Та түр хүлээгээрэй. {dt}-ийн үед үсчин маань сул эсэхийг шалгаж байна...")
    response = requests.get(f"http://localhost:8000/check_conflict?dt={dt}&duration={duration}")
    response_parsed = response.json()
    return response_parsed["message"]

@tool
def ask_user_for_input(user_prompt: str) -> str:
    """Call this function to ask user for input data needed.
    
    Args:
        user_prompt str: a string prompting user to insert the desired information.
    
    Returns:
        str: the answer typed by the user
    """
    user_response = input("\nAI assistant: " + user_prompt + ":\n> Та: ")
    return user_response

@tool
def get_current_datetime() -> str:
    """Use this function to get the current timestamp in ISO format. This tool will help you have the sense of present date and time.
    Returns: the current datetime in ISO format
    """
    return datetime.now().isoformat()


@tool
def connect_to_human_operator() -> str:
    """Use this when you are inquired of the business information you lack."""
    return "FINAL STATE: Сайн байна уу? Та манай кассын ажилтантай холбогдлоо. Танд юугаар туслах вэ?"

@tool
def browse_business_information(question: str) -> str:
    """Use this tool to reply to the question asked by the user

    Args:
        question (str): question regarding the business operations of our barbershop

    Returns:
        str: response to the question if it is found in our documents; otherwise, it says it was not found
    """
    
    if "хуваарь" in question.lower():
        return f"Өглөө 8:00 цагаас оройны 17 цаг хүртэл ажиллана."
    elif "хаяг" in question.lower():
        return f"Манайд ганцхан салбар байгаа, Сүхбаатар дүүрэг Сити Тауэрийн 4 давхарт Чимэгэ салон heh."
    
    return "Response to the question was not found. Connect that person to a human operator, and finish."


def main():
    _ = load_dotenv()
    logging.basicConfig(filename="agent_control.log", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    
    tools = [
        get_current_datetime, 
        ask_user_for_input, 
        check_conflicting_appointment, 
        set_appointment, 
        connect_to_human_operator, 
        browse_business_information
    ]
    
    model = ChatOpenAI(model="gpt-4o-mini", api_key=convert_to_secret_str(OPENAI_KEY))
    
    prompt = """
        You are a barbershop AI agent that communicates in Mongolian language and reason in English.
        
        Your PURPOSE: Help user get appointment at salon using the tools at your disposal.
        
        In your interaction, you MUST follow the mechanism underneath:
        
        1. THOUGHT: you must formulate what action to take and why with respect to your purpose.
        2. THOUGHT: you must specify the function among the bound tools and its arguments to realize that action.
        3. OBSERVATION: you must receive the tool message in response to your function call and update
        your current THOUGHT to identify your current situation and devise the next action.
        4. IMPORTANT to note that you are an LLM node in a LangGraph graph. Your output is strictly required to
        be formatted as an AIMessage object from langchain_core.messages package. To that end, your natural language
        THOUGHT about what to do and why has to be in the `content` property, whilst your specification of
        the functions and the arguments has to populate `tool_calls` property of the AIMessage object returned from your node.
        
        Once appointment process is done, put `FINAL STATE:` string in the AIMessage `content` property with the conclusive message
        to finish the conversation.
        
        Examples:
        * The following AIMessage object is structured in your node as a reply to the user who wanted an appointment.
        
        THOUGHT: I need to start by getting the customer's preferred appointment time to check availability.
        
        tool_calls=[
            {
                'name': 'ask_user_for_input', 
                'args': {
                    'user_prompt': 'Та хэзээ хэдний өдрийн хэдэн цагт үсээ засуулмаар байна?'
                }, 
                'id': 'call_sRSZzRfjYm3cqbRnAOZ3TDAV', 
                'type': 'tool_call'
            }
        ]
        
        * The LangGraph node associated with ACTION invokes ask_user_for_input tool and returns the ensuing ToolMessage object
        which corresponds directly to your OBSERVATION, responsible for updating your next THOUGHT:
        
        ToolMessage(content='Маргааш өглөө 10 цагт засуулах боломжтой юу?', name='ask_user_for_input', tool_call_id='call_sRSZzRfjYm3cqbRnAOZ3TDAV')
        
        For each appointment, the typical :
        1. Get preferred date and time
        2. Get customer's name
        3. Check current datetime
        4. Check for conflicts
        5. Confirm with customer
        6. Set the appointment
        7. Return `FINAL STATE:` with the conclusive message

        Always maintain this THOUGHT/ACTION/OBSERVATION pattern for EVERY step.
        Communicate with users in Mongolian but keep your THOUGHT/ACTION/OBSERVATION in English.
        
        ADDITIONAL REMINDER:
        - Once appointment is set, finish the conversation by including `FINAL STATE:` substring.
        - Never make up responses 
        - Use the provided tools for all interactions.
    """
    
    bot = Agent(model=model, tools=tools, system=prompt)
    
    text_query = input("AI assistant: Сайн байна уу? Та манай үсчинтэй холбогдлоо. Танд яаж туслах вэ?\n\n> Та: ")
    messages = [HumanMessage(content=text_query)]
    results = bot.graph.invoke({"messages" : messages})
    logger.info(results)
    
    
if __name__ == "__main__":
    main()