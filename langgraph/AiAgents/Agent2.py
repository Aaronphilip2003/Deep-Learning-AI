from typing import TypedDict, Union, List
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('api_key'),
    model_name = "llama-3.3-70b-versatile"
)

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print("\nAI:",response.content)
    return state

graph=StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent=graph.compile()
 
conversation_history = []

user_input = input("Human:")
while user_input!="exit":
    conversation_history.append(HumanMessage(content = user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result['messages']
    user_input = input("Human:")

with open("logging.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")
        else:
            f.write(f"{message.content}\n")

print("Conversation history saved to logging.txt")