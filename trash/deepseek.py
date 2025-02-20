import requests
import json

url = "http://localhost:1234/v1/chat/completions"
headers = {"Content-Type": "application/json"}

data = {
    "model": "deepseek-r1-distill-llama-8b",
    "messages": [
        {"role": "system", "content": "Always answer in rhymes. Today is Thursday."},
        {"role": "user", "content": "What day is it today?"}
    ],
    "temperature": 0.7,
    "max_tokens": 100,  # Change from -1 to a valid number
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(response.json())  # Print the API response
except requests.exceptions.RequestException as e:
    print("Request failed:", e)
except json.JSONDecodeError:
    print("Failed to parse JSON response:", response.text)
