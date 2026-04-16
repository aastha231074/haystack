import requests

def main():
    response = requests.post(
       'http://localhost:11434/api/embeddings',
       json={  "model": "mxbai-embed-large","prompt":"Hello, world!"}
    )
    data = response.json()
    print(data)

if __name__ == "__main__":
    main()
