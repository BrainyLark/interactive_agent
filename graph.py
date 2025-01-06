import logging

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

import os
from dotenv import load_dotenv

_ = load_dotenv()

logging.basicConfig(filename="graph_run_3.log", level=logging.INFO)
logger = logging.getLogger("__main__")

@tool
def take_input_from_user(user_input_prompt: Annotated[str, "prompt to display for user to input required info"]) -> str:
    """
    Call this function if you need an additional piece of information from user.
    Return: User input is returned in string form.
    """
    input_data = input(user_input_prompt + ":\n")
    return input_data

@tool
def book_hotel(checkin_name: Annotated[str, "name of person in charge of booking"], 
               number_of_people: Annotated[int, "number of people for whom hotel is booked"],
               city: Annotated[str, "city to book hotel in"], ) -> str:
    
    """
    Book a hotel room for a group of people in a given city under checkin name.
    Return: a string message whether booking succeeded or not.
    """
    
    if city.lower() == "ulaanbaatar" or city.lower() == "ub":
        if number_of_people > 3:
            return "Sorry, we cannot make a reservation for more than 3 people, please provide with a number lower than 4."
        return f"Thank you, you successfully managed to book a hotel for {number_of_people} people. Your checkin name: {checkin_name}."
    return "Sorry we do not have a hotel chain located there."

@tool
def see_city_weather(city: Annotated[str, "name of city you want to see weather in"]) -> str:
    """Call it if you want to see weather of a certain city.
    Return: elaborate string representation of the weather in the given city."""
    if "ub" == city.lower or "ulaanbaatar" == city.lower():
        return "It's 20 degrees and cloudy."
    return "It's 50 degrees and sunny."

tools = [see_city_weather, book_hotel, take_input_from_user]


tool_node = ToolNode(tools=tools)

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    api_key = ""
print(f"api_key={api_key}")
model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", api_key=convert_to_secret_str(api_key)).bind_tools(tools)

# Define the model that determines whether to continue or terminate
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint to agent
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)

workflow.add_edge("tools", "agent")
checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

entry_message = [HumanMessage(content="Book me a hotel in Ulaanbaatar if it's under 30 degrees outside.")]
context = app.invoke({"messages": entry_message}, config={"configurable": {"thread_id": 152}})
logger.info(context)