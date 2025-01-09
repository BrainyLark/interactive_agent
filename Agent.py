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
            if len(llm_message.content) > 0:
                print(f"{llm_message.content}")
                if (type(llm_message.content) is str) and ("final state:" in llm_message.content.lower()):
                     return Decision.END
                print(f"NEXT MOVE: Sending LLM signal to decide actions...\n")        
                return Decision.THINK
            elif len(llm_message.tool_calls) > 0:
                print(f"NEXT MOVE:  LLM chose an action and about to execute...")
                return Decision.ACT
                
        return Decision.END
    
    def act(self, state: AgentState) -> dict | None:
        llm_message = state['messages'][-1]
        
        if not type(llm_message) is AIMessage:
            return
        
        results = []
        for t in llm_message.tool_calls:
            if not t['name'] in self.tools:
                print(f"...bad tool name...\nrequires a retry...")
                result = "bad tool name, retry"
            else:
                print(f"ACT: call function={t['name']} with args={t['args']}")
                result = self.tools[t['name']].invoke(t['args'])
                print(f"OBSERVATION: {str(result)}")
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
            print("COMPLETED ACTION: Sending control to LLM...")
            
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
    
    appointment = { "id": None, "name": name, "operation": None, "created_date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                   "expected_duration": expected_duration, "appointment_datetime": appointment_time, "branch": branch}
    
    print(f"Таны захиалгыг бүртгэлээ. {appointment_time}-д манай {branch}-р салбар дээр уулзацгаая.")
    
    response = requests.post("http://localhost:8000/appointment", json=appointment)
    response_parsed = response.json()
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


def main():
    _ = load_dotenv()
    logging.basicConfig(filename="agent_control.log", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    
    tools = [get_current_datetime, ask_user_for_input, check_conflicting_appointment, set_appointment]
    model = ChatOpenAI(model="gpt-4o-mini", api_key=convert_to_secret_str(OPENAI_KEY))
    
    prompt = """
        You are a barbershop AI agent that communicates in Mongolian language. You MUST follow this EXACT format for every interaction:

        Thought: First, explain your reasoning in English about what you need to do and why.
        Action: Then, specify which tool you'll use and with what parameters.
        Observation: This is the message you receive from the tool invocation or Action
        
        Thought has to be in `content` property of the AIMessage object, which is the output from the LLM.
        Action has to be specified in `tool_calls` property of the AIMessage object, which is the output from the LLM.
        REMEMBER, you HAVE TO explicitly state `FINAL STATE:` in the beginning of AIMessage `content` property
        when the appointment process is completed or you deem it is unnecessary to continue the interaction.
        
        For example:

        Thought: I need to start by getting the customer's preferred appointment time to check availability.
        (This thought has to be found in the following manner AIMessage(content="I need to start by getting..."))
        Action: I will use ask_user_for_input to ask when they want to schedule their appointment.
        (This action decision has to be found in the following manner AIMessage(tool_calls=[ToolCall(...)]))
        [Tool execution happens]
        Observation: Tool responded with 'Next sunday at 7 oclock in the morning'
        ... etc ...
        FINAL STATE: Customer appointment set successfully, we can end the process here.

        For each appointment, you must:
        1. Get preferred date and time
        2. Get customer's name
        3. Check current datetime
        4. Check for conflicts
        5. Confirm with customer
        6. Set the appointment
        7. Send FINAL STATE: in the AIMessage content with the conclusive message.

        Always maintain this Thought/Action/Observation pattern for EVERY step.
        Communicate with users in Mongolian but keep your Thought/Action/Observation in English.

        Never make up responses - use the provided tools for all interactions.
    """
    
    bot = Agent(model, tools, prompt)
    
    text_query = input("AI assistant: Сайн байна уу? Та манай үсчинтэй холбогдлоо. Танд яаж туслах вэ?\n\n> Та: ")
    messages = [HumanMessage(content=text_query)]
    results = bot.graph.invoke({"messages" : messages})
    logger.info(results)
    
    
if __name__ == "__main__":
    main()