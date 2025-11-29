# 🦊 YAE-AI Project

> _Build your own AI-powered VTuber assistant_ ✨

---

## 🌟 Overview

**YAE-AI** is your gateway to self-hosting powerful AI VTubers with style. Whether you're streaming, coding, or just vibing with your AI waifu, this project lets you run vLLMs (very Large Language Models) locally using Docker, while keeping the flexibility to switch between providers like ChatGPT, Claude, and Gemini.

No cap, this is the ultimate foundation for your AI companion dreams. 💜✨

> 🎭 **Pro tip**: Perfect for VTuber streams, dev assistants, or just having a based AI friend who gets your references.

---

## ✨ Features

- 🐳 **Docker-powered vLLM hosting** — Clean, isolated, professional setup
- 🔄 **Multi-provider support** — ChatGPT, Claude, Gemini? We got you
- 🎯 **Character roleplay ready** — Built for immersive VTuber interactions
- ⚡ **UV package manager** — Lightning-fast Python dependency handling
- 🎨 **Fully customizable** — Make your AI assistant uniquely yours

---

## 🛠️ Prerequisites

Before we pop off, make sure you've got these installed:

| Tool | Description | Installation |
|------|-------------|--------------|
| **Python 3.10+** | The language of the gods 🐍 | [Download](https://www.python.org/downloads/) |
| **UV** | Next-gen Python package manager | [Install UV](https://docs.astral.sh/uv/getting-started/installation/) |
| **Docker** | Container platform for vLLMs 🐳 | [Get Docker](https://docs.docker.com/get-docker/) |
| **Git** | For cloning the repo | [Install Git](https://git-scm.com/downloads) |

> ⚠️ **Important**: Make sure Docker Desktop is running before you start!

---

## 💾 Installation

Let's get this bread with these quick commands:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/OuOSama/YAE-AI.git yae-ai
cd yae-ai
```

### 2️⃣ Install Dependencies

```bash
uv sync
```

This command pulls all the Python packages you need. UV is built different — it's fast AF. ⚡

### 3️⃣ Download the Model

```bash
uv run src/utils/save.py
```

This script fetches your AI model from Hugging Face and saves it locally. Patience is key here, bestie — models can be thicc. 📦

### 4️⃣ Launch vLLM Containers

```bash
docker compose up -d
```

Your vLLM server is now running in the background. We're so back. 🎯

---

## 🚀 Usage

Time to bring your AI to life! Here's how you interact with your model:

```python
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
            "content": "Where is Raiden?"
        }
    ],
    temperature=0.2,
    max_tokens=256
)

print(response.choices[0].message.content)
```

### 🎮 Advanced Configuration

| Parameter | Purpose | Recommended Values |
|-----------|---------|-------------------|
| `model` | Model path in Docker | `"/model"` |
| `temperature` | Response creativity | `0.2-0.4` (focused) <br> `0.7-1.0` (creative) |
| `max_tokens` | Response length limit | `256-1024` |
| `top_p` | Nucleus sampling | `0.9-1.0` |
| `frequency_penalty` | Reduce repetition | `0.0-0.5` |

---

## 🎨 Character System Prompt Examples

### VTuber Personality

```python
system_prompt = """
You are Kitsune-chan, a playful fox VTuber who loves coding streams and anime.
You speak with Gen Z energy, use developer slang naturally, and occasionally 
reference anime or gaming culture. You're supportive but also tease viewers 
with wit. Never break character or mention being an AI.
"""
```

### Coding Assistant

```python
system_prompt = """
You are a senior dev with VTuber aesthetics. Explain code concepts clearly,
use analogies from anime/games when helpful, and keep responses encouraging.
You're the cool senpai who makes programming feel less scary.
"""
```

---

## 🔧 Troubleshooting

### Docker Issues

```bash
# Check if Docker is running
docker ps

# Restart containers if needed
docker compose restart

# View logs for debugging
docker compose logs -f
```

### Model Loading Errors

- **Issue**: Model download fails
  - **Fix**: Check your internet connection and Hugging Face access
  - **Fix**: Verify you have enough disk space (models are large!)

- **Issue**: vLLM server won't start
  - **Fix**: Ensure ports 8000-8001 aren't already in use
  - **Fix**: Check Docker has enough RAM allocated (8GB+ recommended)

---

## 🌸 Contributing

Want to make YAE-AI even more bussin'? We'd love your contributions! 

- 🐛 Report bugs via [GitHub Issues](https://github.com/OuOSama/YAE-AI/issues)
- 💡 Suggest features or improvements
- 🔧 Submit pull requests with enhancements
- 🎨 Share your character system prompts!

### Development Setup

```bash
# Fork the repo, then clone your fork
git clone https://github.com/YOUR-USERNAME/YAE-AI.git
cd YAE-AI

# Create a feature branch
git checkout -b feature/your-cool-feature

# Make changes, commit, and push
git add .
git commit -m "feat: add cool feature"
git push origin feature/your-cool-feature
```

---

## 📊 Project Structure

```
yae-ai/
├── src/
│   ├── utils/
│   │   └── save.py          # Model downloader
│   └── main.py              # Main application
├── docker-compose.yml       # vLLM container config
├── pyproject.toml           # UV dependencies
├── README.md                # You are here!
└── LICENSE                  # MIT License
```

---

## 🌈 Roadmap

- [ ] Voice synthesis integration (TTS)
- [ ] WebUI for easier configuration
- [ ] Multi-language support
- [ ] Fine-tuning scripts for custom models
- [ ] Live2D integration for visual VTuber avatar
- [ ] Streaming platform integrations (Twitch, YouTube)

---

## 💖 Acknowledgments

Shoutout to the legends who made this possible:

- [vLLM Team](https://github.com/vllm-project/vllm) for the amazing inference engine
- [Astral](https://astral.sh/) for creating UV
- The VTuber community for endless inspiration
- All contributors and supporters ✨

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

You're free to use, modify, and distribute this project. Just don't be cringe about it. 💜

---

## 🦊 Credits

**Created with love by [OuOSama](https://github.com/OuOSama)**

If this project helped you, consider:
- ⭐ Starring the repo
- 🔄 Sharing with friends
- ☕ [Buying me a coffee](https://ko-fi.com/ouosama) (if you're feeling generous!)

---

<div align="center">

**Made with 💜 for the VTuber and developer community**

*Stay based, stay creative* ✨

</div>
