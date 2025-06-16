from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
from langgraph.graph.message import add_messages

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a+b

@tool
def subtract(a: int, b: int):
    """This is a subtraction function that subtracts two numbers"""
    return a-b

@tool
def product(a: int,b: int):
    """This function multiplies two numbers"""
    return a*b

@tool
def divide(a:int, b: int):
    """This function divides two numbers provided that the divisor is not 0"""
    return a/b

tools = [add, subtract, product, divide]

model = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('api_key'),
    model_name = "llama-3.3-70b-versatile"
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my queries to the best of your ability.")
    response = model.invoke([system_prompt]+state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools",tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges("our_agent", should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream) :
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages":[("user","Add 3 and 4, after that add 3 to that result and then subtract 9 from that then multiply it by 100 and divide it by 2 ")]}
print_stream(app.stream(inputs,stream_mode="values"))
