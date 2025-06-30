from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()

def read_file(path):
    text=""
    with open (path) as f:
        text=f.read()
    return text

def readme_generator():
    text=read_file("./transcript.txt")
    llm=ChatGroq(temperature=0,
    groq_api_key = os.getenv("api_key"),
    model_name = "llama-3.3-70b-versatile")
    prompt=f"You are a professor of computer science in a university, explain the text to the best of your ability and make good notes so that i wont have to ever read the text again and generate a markdown file. This is the text {text} "

    answer = llm.invoke(prompt)

    with open ("./answer.md","w+") as f:
        f.write(str(answer.content))

readme_generator()