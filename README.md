## 📦 1. Install Dependencies

### **Python 3.8+ recommended**

Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux

pip install streamlit streamlit-audiorecorder pydub faster-whisper
brew install ffmpeg

## 📦 2. Run the Project
streamlit run app.py


### 3. Integration with Cerebras
pip install --upgrade cerebras_cloud_sdk
export CEREBRAS_API_KEY="your-cerebras-key"
streamlit run new-app.py

