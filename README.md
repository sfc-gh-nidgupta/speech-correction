# ❄️ Snowflake Voice Assistant

A voice-powered assistant for Snowflake questions. Record your question, get it transcribed, enhanced, and answered using AI.

---

## 📦 1. Install Dependencies

### Prerequisites
- **Python 3.11** (recommended for best compatibility)
- **ffmpeg** (for audio processing)
- **Cerebras API Key** (for AI features)

### Setup

```bash
# Clone the repository
git clone https://github.com/sfc-gh-nidgupta/speech-correction.git
cd speech-correction

# Create virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux

# Install Python dependencies
pip install streamlit streamlit-audiorecorder pydub faster-whisper
pip install cerebras_cloud_sdk

# Install ffmpeg (macOS)
brew install ffmpeg
```

---

## 🚀 2. Run the App

### Basic Mode (Transcription + Cleanup)
```bash
source .venv/bin/activate
export CEREBRAS_API_KEY="your-cerebras-key"
streamlit run app.py
```

### Advanced Mode (Transcription + Enhancement + Snowflake Agent)
```bash
source .venv/bin/activate
export CEREBRAS_API_KEY="your-cerebras-key"
streamlit run new-app.py
```

Open http://localhost:8501 in your browser.

---

## 🔬 Advanced Mode Features

The Advanced Mode (`new-app.py`) provides a complete voice-to-answer pipeline:

| Step | Description |
|------|-------------|
| 🎙️ **Record** | Capture your Snowflake question via voice |
| 📝 **Transcribe** | Local Whisper transcription |
| ✨ **Enhance** | AI cleans up and enriches the query with proper Snowflake terminology |
| ❄️ **Answer** | Snowflake Agent answers from the FAQ knowledge base |

### Toggle in Sidebar
- **Advanced Mode ON**: Full pipeline with query enhancement
- **Advanced Mode OFF**: Direct transcription to agent (skip enhancement)

---

## 📁 Project Structure

```
speech-correction/
├── app.py              # Basic transcription + cleanup
├── new-app.py          # Advanced mode with Snowflake Agent
├── snowflake_agent.py  # LLM-powered Snowflake Q&A module
├── FAQ.md              # Snowflake terminology knowledge base
└── README.md
```

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `CEREBRAS_API_KEY` | Your Cerebras API key for LLM features |

---

## 💡 Example Questions

Try asking these via voice:

- "What is a virtual warehouse?"
- "Explain Time Travel in Snowflake"
- "What's the difference between streams and tasks?"
- "How does zero-copy cloning work?"
- "What is Snowflake Cortex AI?"

---

## 🛠️ Troubleshooting

**App won't start?**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate

# Check if dependencies are installed
pip list | grep streamlit
```

**Audio not recording?**
- Allow microphone access in your browser
- Make sure ffmpeg is installed: `brew install ffmpeg`

**API errors?**
- Verify your Cerebras API key is set: `echo $CEREBRAS_API_KEY`
