from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
    
def node_a(state: State):
    return {"foo": "a", "bar" : ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar" : ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = { "configurable" : { "thread_id" : "1" } }
graph.invoke({"foo": ""}, config)
print(graph.get_state(config))