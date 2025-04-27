import requests

def main():
    proxy_url = "http://localhost:8000/chat"

    user_question = "How does the server start and accept secure connections?"

    payload = {
        "messages": [
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.2
    }

    response = requests.post(proxy_url, json=payload)

    if response.status_code == 200:
        data = response.json()
        try:
            print("\n=== Drafted Answer ===\n")
            print(data["choices"][0]["message"]["content"])
        except Exception:
            print("Response structure unexpected:", data)
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()