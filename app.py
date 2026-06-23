from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load .env
load_dotenv()

# Check API Key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found!")

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

# State
class State(TypedDict):
    question: str
    answer: str

# Node
def chatbot(state):
    response = llm.invoke(state["question"])
    return {"answer": response.content}

# Graph
graph = StateGraph(State)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# Compile
app = graph.compile()

print("🤖 LangGraph Gemini Chatbot Started")
print("Type 'exit' to quit\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        break

    try:
        result = app.invoke({"question": question})
        print("\nAI:", result["answer"])
        print()
    except Exception as e:
        print("\nError:", e)