from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_API_KEY"),  # This is the default and can be omitted
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o",
)

print(chat_completion)