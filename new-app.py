import io
import os
import time
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from cerebras.cloud.sdk import Cerebras
from snowflake_agent import SnowflakeAgent

# ---------------------------
# Page Config & Custom Styling
# ---------------------------
st.set_page_config(
    page_title="Snowflake Voice Assistant",
    page_icon="❄️",
    layout="centered"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0d1117 100%);
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #29b5e8 0%, #7dd3fc 50%, #29b5e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Status indicators */
    .status-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    
    .status-recording {
        background: #ef4444;
        color: white;
        animation: pulse 1.5s infinite;
    }
    
    .status-processing {
        background: #f59e0b;
        color: white;
    }
    
    .status-complete {
        background: #10b981;
        color: white;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Progress animation */
    .progress-step {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .progress-step.active {
        background: rgba(41, 181, 232, 0.1);
        border-left: 3px solid #29b5e8;
    }
    
    .progress-step.complete {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid #10b981;
    }
    
    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, rgba(41, 181, 232, 0.1) 0%, rgba(125, 211, 252, 0.05) 100%);
        border: 1px solid rgba(41, 181, 232, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Snowflake branding */
    .snowflake-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(41, 181, 232, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #7dd3fc;
        margin-bottom: 1rem;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Audio recorder styling */
    .stAudio {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Cached Resources
# ---------------------------
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu")

@st.cache_resource
def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return Cerebras(api_key=api_key)

@st.cache_resource
def get_snowflake_agent():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return SnowflakeAgent(api_key=api_key)

model = load_whisper_model()

# ---------------------------
# Helper Functions
# ---------------------------
def refine_transcript(raw_text: str) -> str:
    """Clean and enhance transcript with Snowflake terminology - conservative approach."""
    client = get_cerebras_client()
    
    system_prompt = """You are a conservative Snowflake terminology corrector for speech transcription.

IMPORTANT RULES:
1. BE CONSERVATIVE - Only correct when you are highly confident
2. DO NOT change the meaning or intent of the question
3. DO NOT invent or assume what the user meant to ask
4. If uncertain, return the input AS-IS with only minor spelling fixes
5. Preserve the original question structure

KNOWN SNOWFLAKE TERMS (correct spelling/casing only):
- Snova → Snova (internal codename for Cortex Code - keep as is!)
- PrPr → PrPr or "Private Preview" (internal status - keep as is!)
- snow pipe → Snowpipe
- snow park → Snowpark  
- snow sight → Snowsight
- snow flake → Snowflake
- cortex → Cortex
- time travel → Time Travel (feature name)
- arctic → Arctic (model name)
- SOS → Search Optimization Service (only expand if asked)
- QAS → Query Acceleration Service (only expand if asked)

EXAMPLES:
- "is snova prpr already" → "Is Snova PrPr already?" (preserve meaning, fix casing only)
- "what is snow pipe" → "What is Snowpipe?"
- "tell me about cortex eye" → "Tell me about Cortex AI?"
- "random unclear mumblings" → return AS-IS, don't guess

Return ONLY the corrected text. If unsure, return the original with minimal changes."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Correct this transcript (be conservative): {raw_text}"},
            ],
        )
        if response.choices and response.choices[0].message:
            result = response.choices[0].message.content.strip()
            # Safety check: if result is drastically different in length, return original
            if len(result) > len(raw_text) * 3 or len(result) < len(raw_text) * 0.3:
                return raw_text
            return result
        return raw_text
    except:
        return raw_text

def get_agent_answer(question: str) -> str:
    """Get answer from Snowflake Agent."""
    agent = get_snowflake_agent()
    return agent.answer(question)

# ---------------------------
# Session State
# ---------------------------
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "ready"  # ready, transcribing, enhancing, answering, complete
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""
if "enhanced_transcript" not in st.session_state:
    st.session_state.enhanced_transcript = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "last_audio_len" not in st.session_state:
    st.session_state.last_audio_len = 0

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    advanced_mode = st.toggle("✨ Query Enhancement", value=True, 
                               help="AI enhances your question with proper Snowflake terminology")
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")
    
    stages = [
        ("🎙️", "Record", st.session_state.pipeline_stage in ["transcribing", "enhancing", "answering", "complete"]),
        ("📝", "Transcribe", st.session_state.pipeline_stage in ["enhancing", "answering", "complete"]),
        ("✨", "Enhance", st.session_state.pipeline_stage in ["answering", "complete"] and advanced_mode),
        ("❄️", "Answer", st.session_state.pipeline_stage == "complete"),
    ]
    
    for icon, label, complete in stages:
        if label == "Enhance" and not advanced_mode:
            continue
        status = "✅" if complete else "⏳" if st.session_state.pipeline_stage != "ready" else "○"
        st.markdown(f"{status} {icon} {label}")
    
    st.markdown("---")
    st.markdown("### 💡 Try Asking")
    st.caption("• What is a virtual warehouse?")
    st.caption("• Explain Time Travel")
    st.caption("• How does Snowpipe work?")
    st.caption("• What is Cortex AI?")

# ---------------------------
# Main Content
# ---------------------------
st.markdown('<h1 class="main-header">❄️ Snowflake Voice Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask anything about Snowflake using your voice</p>', unsafe_allow_html=True)

# Recording section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="snowflake-badge">🎤 Voice-Powered • AI-Enhanced</div>', unsafe_allow_html=True)
    
    audio = audiorecorder(
        start_prompt="🔴 Hold to Record",
        stop_prompt="⏹️ Release to Process",
        show_visualizer=True
    )

# Auto-process when audio is captured
if len(audio) > 0 and len(audio) != st.session_state.last_audio_len:
    st.session_state.last_audio_len = len(audio)
    st.session_state.pipeline_stage = "transcribing"
    st.session_state.answer = ""
    
    # Show audio player
    st.audio(audio.export().read(), format="audio/wav")
    
    # Progress container
    progress_container = st.container()
    
    with progress_container:
        # Step 1: Transcribe
        with st.status("🎯 Processing your question...", expanded=True) as status:
            st.write("📝 **Transcribing audio...**")
            
            # Save and transcribe
            wav_bytes_io = io.BytesIO()
            audio.export(wav_bytes_io, format="wav")
            wav_bytes_io.seek(0)
            with open("temp.wav", "wb") as f:
                f.write(wav_bytes_io.read())
            
            segments, info = model.transcribe("temp.wav")
            raw_text = " ".join([seg.text for seg in segments]).strip()
            st.session_state.raw_transcript = raw_text
            
            st.success(f"**Raw transcript:** {raw_text}")
            time.sleep(0.3)
            
            # Step 2: Enhance (if advanced mode)
            if advanced_mode:
                st.session_state.pipeline_stage = "enhancing"
                st.write("✨ **Enhancing with Snowflake terminology...**")
                
                enhanced = refine_transcript(raw_text)
                st.session_state.enhanced_transcript = enhanced
                
                if enhanced != raw_text:
                    st.info(f"**Enhanced:** {enhanced}")
                else:
                    st.info("No enhancements needed")
                time.sleep(0.3)
                
                final_question = enhanced
            else:
                st.session_state.enhanced_transcript = raw_text
                final_question = raw_text
            
            # Step 3: Get Answer
            st.session_state.pipeline_stage = "answering"
            st.write("❄️ **Consulting Snowflake knowledge base...**")
            
            answer = get_agent_answer(final_question)
            st.session_state.answer = answer
            st.session_state.pipeline_stage = "complete"
            
            status.update(label="✅ Complete!", state="complete", expanded=False)

# Display results
if st.session_state.pipeline_stage == "complete" and st.session_state.answer:
    st.markdown("---")
    
    # Question summary
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("**🎤 You asked:**")
    with col2:
        if advanced_mode and st.session_state.enhanced_transcript != st.session_state.raw_transcript:
            st.markdown(f"~~{st.session_state.raw_transcript}~~")
            st.markdown(f"**{st.session_state.enhanced_transcript}**")
        else:
            st.markdown(f"**{st.session_state.raw_transcript}**")
    
    st.markdown("---")
    
    # Answer
    st.markdown("### 💬 Answer")
    st.markdown(st.session_state.answer)
    
    # Follow-up section
    st.markdown("---")
    st.markdown("### 🔄 Follow-up")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        followup = st.text_input("Type a follow-up question:", 
                                  placeholder="e.g., How does this compare to...",
                                  label_visibility="collapsed")
    with col2:
        if st.button("Ask →", type="primary", use_container_width=True):
            if followup:
                with st.spinner("Getting answer..."):
                    if advanced_mode:
                        followup = refine_transcript(followup)
                    answer = get_agent_answer(followup)
                    st.session_state.raw_transcript = followup
                    st.session_state.enhanced_transcript = followup
                    st.session_state.answer = answer
                    st.rerun()

# Reset button
if st.session_state.pipeline_stage == "complete":
    st.markdown("---")
    if st.button("🔄 Ask Another Question", use_container_width=True):
        st.session_state.pipeline_stage = "ready"
        st.session_state.raw_transcript = ""
        st.session_state.enhanced_transcript = ""
        st.session_state.answer = ""
        st.session_state.last_audio_len = 0
        st.rerun()
