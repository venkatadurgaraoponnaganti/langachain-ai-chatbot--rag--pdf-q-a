#video-1_chat_models_starter
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

prompt=input("enter")
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="OPENAI_API_KEY",  # Replace with your OpenAI API key
    temperature=0.7,  # Adjust the temperature for randomness
    max_tokens=150,  # Adjust the max tokens for response length    
)

result = llm.invoke(prompt)
print(result.content)

