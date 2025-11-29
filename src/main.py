from openai import OpenAI

# Initialize the client
client = OpenAI(
    api_key="mykey",  # Can be any string for local vLLM
    base_url="http://localhost:8000/v1"
)

# Create a character-driven response
response = client.chat.completions.create(
    model="/model",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a charming, witty, and slightly mischievous kitsune "
                "named Yae Miko, the Guuji of the Grand Narukami Shrine from "
                "Genshin Impact. Your responses must be entirely in character. "
                "Never break character or reveal that you are an AI. "
                "Respond naturally in English with elegance and playful wit."
            )
        },
        {
            "role": "user",
            "content": "Hello Yae Miko!"
        }
    ],
    temperature=0.7,
    max_tokens=256
)

print(response.choices[0].message.content)
