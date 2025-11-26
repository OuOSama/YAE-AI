# 🦊 YAE-AI Project

> _Build your own AI-powered VTuber assistant_ ✨

---

## 📖 Overview

**YAE-AI** is a Python-based project that empowers you to self-host your own vLLMs (very Large Language Models) using modern tooling like Docker and the UV package manager, while still allowing you to easily switch to other providers such as ChatGPT, Claude, Gemini, and more 🦊✨.

Whether you're building a virtual assistant, streaming companion, or interactive AI waifu, this setup gives you the foundation to make it happen. 💜

---

## 🛠️ Prerequisites

Before diving in, make sure you have these tools ready:

| Tool       | Description                             | Link                                                                          |
| ---------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| **UV**     | Modern Python package manager 🐍        | [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/) |
| **Docker** | Container platform for running vLLMs 🐳 | [Get Docker](https://docs.docker.com/)                                        |

> ⚠️ **Important**: Ensure your Docker is running and accessible from your terminal before proceeding!

---

## 💾 Installation

Get up and running in just 3 commands:

### Step 1: Sync Python Dependencies

```
uv sync
```

### Step 2: load model from hugging face to our folder

```
uv run src/utils/save.py
```

### Step 3: Start vLLM Containers

```bash
docker compose up -d
```

This spins up your vLLM containers in detached mode, ready to serve your AI models. 🎯

---

## 🚀 Usage

Once your environment is set up, here's how to interact with your AI model:

```python
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

```

### 🎨 Configuration Options

| Parameter     | Description                            | Example                      |
| ------------- | -------------------------------------- | ---------------------------- |
| `model`       | The model name served by Docker        | `"/model"`                 |
| `temperature` | Controls response randomness (0.0-1.0) | `0.2` for focused responses  |
| `base_url`    | vLLM server endpoint                   | `"http://localhost:8000/v1"` |

---

## 💡 Tips & Best Practices

- **Temperature Settings**: Lower values (0.2-0.4) for consistent character roleplay, higher values (0.7-1.0) for creative responses
- **System Prompts**: Craft detailed character descriptions to maintain immersive interactions
- **Model Selection**: Choose models that align with your performance needs and character complexity

---

## 🌸 Contributing

Want to make YAE-AI even better? Contributions are welcome! Feel free to:

- Report bugs 🐛
- Suggest features 💡
- Submit pull requests 🔧
- Share your custom character prompts 🎨

---

## 📜 License

---
