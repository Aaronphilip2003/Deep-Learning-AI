from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGroq(
    temperature=0,
    groq_api_key = os.getenv("api_key"),
    model_name = "llama-3.3-70b-versatile"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    print("AI: ", response.content)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

input_message = input("Enter a message: ")
while input_message.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=input_message)]})
    input_message = input("Enter a message: ")

print("Exiting...")