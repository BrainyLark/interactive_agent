import logging
import requests
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import os
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(filename="appointment_app_run_002.log", level=logging.INFO)
logger = logging.getLogger("__main__")

_ = load_dotenv()

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
    user_response = input(user_prompt + ":\n\n")
    return user_response

@tool
def get_current_datetime() -> str:
    """Use this function to get the current timestamp in ISO format.
    Returns: the current datetime in ISO format
    """
    return datetime.now().isoformat()

api_key = os.getenv("OPENAI_API_KEY", "")
tools = [get_current_datetime, ask_user_for_input, set_appointment, check_conflicting_appointment]
tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o", api_key=convert_to_secret_str(api_key)).bind_tools(tools)

def should_continue(state: MessagesState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

system_prompt = """
    You are a barbershop AI agent that makes haircut appointments for users. And you must communicate in
    Mongolian language. You MUST use the following tools to assist users:
    - get_current_datetime: People don't express the reservation datetime in full format, so you have to use this tool to guess the unambiguous datetime.
    - ask_user_for_input: Use this to get necessary information from the user (you must let the user confirm the suggested time and ask again)
    - check_conflicting_appointment: Use this to verify if the requested time slot is available
    - set_appointment: Use this to create the final appointment

    For each appointment, you need to:
    1. Gather necessary information using ask_user_for_input (name, preferred date and time)
    2. Use `get_current_datetime` for the present datetime and suppose the accurate datetime from the user's input. You have to have it confirmed by the user again.
    3. Check if the time slot is available using check_conflicting_appointment (duration is 45 minutes)
    4. Remember if there is a conflict in appointment, you must let the user confirm the suggested time and ask again using `ask_user_for_input` tool.
    5. Create the appointment using set_appointment

    Always use these tools to complete the task. Do not provide fictional responses.
"""

def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

entry_message = [SystemMessage(content=system_prompt)]
context = app.invoke({"messages": entry_message}, config={"configurable": {"thread_id": 1004}})

logger.info(context)