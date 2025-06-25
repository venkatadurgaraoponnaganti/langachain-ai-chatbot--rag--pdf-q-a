# 3.chat model _alternative models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="AIzaSyD3Pj8Y7g0B7MhYBAPhkgWqSEYOmzijGIE"  # Make sure this key is active
)

messages = [
    SystemMessage(content="You are an expert in social media content strategy"),
    HumanMessage(content="Give a short tip to create engaging posts on Instagram")
]

response = llm.invoke(messages)
print(response.content)
