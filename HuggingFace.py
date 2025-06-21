import requests
import json

# Constants
API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
API_KEY = "************************************"  # You can also load from environment for security

# Get user input
user_query = input("Ask something: ")

# Prepare request payload
payload = {
    "messages": [
        {
            "role": "user",
            "content": user_query
        }
    ],
    "model": "minimaxai/minimax-m1-80k",
    "stream": False
}

# Send POST request
response = requests.post(
    API_URL,
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    data=json.dumps(payload)
)

# Parse and print response
if response.status_code == 200:
    result = response.json()
    try:
        content = result["choices"][0]["message"]["content"]
        print("\nAI:", content)
    except (KeyError, IndexError):
        print("Unexpected response format:", result)
else:
    print("Error:", response.status_code, response.text)
