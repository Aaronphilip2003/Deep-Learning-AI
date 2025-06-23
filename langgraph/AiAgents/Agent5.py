from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv('api_key'),
    model_name = "llama-3.3-70b-versatile"
)
