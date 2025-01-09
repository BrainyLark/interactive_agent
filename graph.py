import logging
import requests
from typing import Literal, List

from langchain_core.messages import SystemMessage, AIMessage

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import os
from datetime import datetime
from dotenv import load_dotenv

logging.basicConfig(filename="appointment_app_run_003.log", level=logging.INFO)
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
    user_response = input("\nAI Assistant: " + user_prompt + ":\n> User: ")
    return user_response

@tool
def get_current_datetime() -> str:
    """Use this function to get the current timestamp in ISO format. This tool will help you have the sense of present date and time.
    Returns: the current datetime in ISO format
    """
    return datetime.now().isoformat()

api_key = os.getenv("OPENAI_API_KEY", "")
tools = [get_current_datetime, ask_user_for_input, set_appointment, check_conflicting_appointment]
tool_node = ToolNode(tools)
        

model = ChatOpenAI(model="gpt-4o-mini", api_key=convert_to_secret_str(api_key)).bind_tools(tools)

def should_continue(state: MessagesState) -> str:
    messages = state['messages']
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

system_prompt = """
    You are a barbershop AI agent that communicates in Mongolian language. You MUST follow this EXACT format for every interaction:

    Thought: First, explain your reasoning in English about what you need to do and why.
    Action: Then, specify which tool you'll use and with what parameters.
    Observation: Wait for the tool response.

    For example:

    Thought: I need to start by getting the customer's preferred appointment time to check availability.
    Action: I will use ask_user_for_input to ask when they want to schedule their appointment.
    [Tool execution happens]
    Observation: [System will provide the observation]

    For each appointment, you must:
    1. Get preferred date and time
    2. Get customer's name
    3. Check current datetime
    4. Check for conflicts
    5. Confirm with customer
    6. Set the appointment

    Always maintain this Thought/Action/Observation pattern for EVERY step.
    Communicate with users in Mongolian but keep your Thought/Action/Observation in English.

    Never make up responses - use the provided tools for all interactions.
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
context = app.invoke({"messages": entry_message}, config={"configurable": {"thread_id": 124563}})

logger.info(context)