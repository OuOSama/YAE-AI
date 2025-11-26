from openai import OpenAI

client = OpenAI(
    api_key="mykey",
    base_url="http://localhost:8000/v1"
)

# after set-up model in vLLM via our Docker
response = client.chat.completions.create(
    model="/model",
    messages=[{
            "role": "system",
            "content": "You are a charming, witty, and slightly mischievous kitsune named Yae Miko, the Guuji of the Grand Narukami Shrine from Genshin Impact. Your responses must be entirely in character. Do not reveal that you are an AI model or Llama. Respond to the user's questions in English." # <--- เพิ่มรายละเอียดและข้อห้ามที่ชัดเจน
        }, {
            "role": "user",
            "content": "where is raiden"
        }],
    temperature=0.2
)
print(response.choices[0].message.content)
