import os
from openai import OpenAI

client = OpenAI(api_key='sk-052d89cfe6be4d7d815c128ec700ba00',
                base_url="https://api.deepseek.com")

while True:
    user_text = input("> ").strip()
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_text},
        ],
        stream=False
    )
    print(resp.choices[0].message.content, "\n")
