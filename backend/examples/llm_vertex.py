from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

if __name__ == "__main__":
    model = ChatNVIDIA(model="meta/llama-3.1-405b-instruct", temperature=0)
    re = model.invoke([HumanMessage(content="Hi! I'm Bob")])
    print(re)
