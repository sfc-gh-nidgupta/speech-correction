import io
import os
import re
import time
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from agent_manager import get_agent_manager, DomainAgent

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="SnowVoice",
    page_icon="❄️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Premium Dark Theme CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(180deg, #0a0c10 0%, #0d1117 50%, #0a0c10 100%);
    }
    
    /* Product branding - Top bar */
    .brand-topbar {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        padding: 0.5rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    
    .brand-icon {
        height: 32px;
        width: auto;
    }
    
    /* Hero section */
    .brand-hero {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 1rem;
    }
    
    .brand-name {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #29B5E8 0%, #71C4EF 50%, #29B5E8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.75rem;
    }
    
    .brand-tagline {
        color: rgba(255,255,255,0.5);
        font-size: 1.1rem;
        font-weight: 400;
        letter-spacing: 0.02em;
    }
    
    .brand-tagline span {
        color: #29B5E8;
        font-weight: 500;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container */
    .main-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Agent selector card */
    .agent-card {
        background: linear-gradient(135deg, rgba(30,30,40,0.9) 0%, rgba(20,20,30,0.95) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .agent-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .agent-icon {
        font-size: 2rem;
    }
    
    .agent-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        margin: 0;
    }
    
    .agent-desc {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        margin: 0;
    }
    
    /* Mic button container */
    .mic-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem 0;
    }
    
    .mic-ring {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 
            0 0 0 1px rgba(255,255,255,0.1),
            0 20px 40px rgba(0,0,0,0.4),
            inset 0 1px 0 rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .mic-ring:hover {
        transform: scale(1.05);
        box-shadow: 
            0 0 0 1px rgba(99,102,241,0.3),
            0 0 30px rgba(99,102,241,0.2),
            0 20px 40px rgba(0,0,0,0.4);
    }
    
    .mic-hint {
        color: rgba(255,255,255,0.4);
        font-size: 0.8rem;
        margin-top: 1rem;
        text-align: center;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, rgba(30,30,40,0.9) 0%, rgba(20,20,30,0.95) 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .result-label {
        color: rgba(255,255,255,0.4);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    
    .result-question {
        color: #fff;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    
    .result-answer {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        line-height: 1.7;
    }
    
    /* Correction indicator */
    .correction-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(99,102,241,0.15);
        color: #818cf8;
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.7rem;
        margin-bottom: 0.5rem;
    }
    
    /* View files button */
    .view-files-btn {
        color: rgba(255,255,255,0.4);
        font-size: 0.75rem;
        cursor: pointer;
        transition: color 0.2s;
    }
    
    .view-files-btn:hover {
        color: rgba(255,255,255,0.7);
    }
    
    /* Streamlit overrides */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4);
    }
    
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        height: 40px;
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Cached Resources
# ---------------------------
@st.cache_resource
def load_whisper_model():
    return WhisperModel("small", device="cpu")

model = load_whisper_model()
agent_manager = get_agent_manager()

# ---------------------------
# Session State
# ---------------------------
defaults = {
    "pipeline_stage": "ready",
    "raw_transcript": "",
    "enhanced_transcript": "",
    "answer": "",
    "last_audio_len": 0,
    "show_files": False,
    "advanced_mode": True
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------
# Get Agents
# ---------------------------
agents = agent_manager.list_agents()

if not agents:
    st.error("No agents found. Please add agents to the `agents/` directory.")
    st.stop()

# ---------------------------
# Product Branding
# ---------------------------
st.markdown("""
<div class="brand-topbar">
    <img class="brand-icon" src="https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg" alt="Snowflake" onerror="this.src='https://companieslogo.com/img/orig/SNOW-35164165.png'">
</div>
<div class="brand-hero">
    <div class="brand-name">SnowVoice</div>
    <p class="brand-tagline">Real Voice Intelligence, Powered by <span>Your Snowflake Data</span></p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Agent Selection (Top)
# ---------------------------
agent_options = {f"{a.icon} {a.name}": a for a in agents}
col1, col2 = st.columns([3, 1])

with col1:
    selected_name = st.selectbox(
        "Agent",
        options=list(agent_options.keys()),
        label_visibility="collapsed"
    )

with col2:
    if st.button("📄 View", use_container_width=True, help="View agent files"):
        st.session_state.show_files = not st.session_state.show_files

selected_agent = agent_options[selected_name]

# Show agent files if toggled
if st.session_state.show_files:
    with st.expander(f"📁 {selected_agent.name} Agent Files", expanded=True):
        tab1, tab2 = st.tabs(["📋 Terms (YAML)", "📖 Knowledge (MD)"])
        
        with tab1:
            terms_content = selected_agent.load_terms()
            st.code(terms_content[:3000] + ("..." if len(terms_content) > 3000 else ""), language="yaml")
        
        with tab2:
            knowledge_content = selected_agent.load_knowledge()
            st.markdown(knowledge_content[:5000] + ("..." if len(knowledge_content) > 5000 else ""))

# ---------------------------
# Agent Header
# ---------------------------
st.markdown(f"""
<div class="agent-card">
    <div class="agent-header">
        <span class="agent-icon">{selected_agent.icon}</span>
        <div>
            <h2 class="agent-name">{selected_agent.name}</h2>
            <p class="agent-desc">{selected_agent.description}</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Settings Row
# ---------------------------
col1, col2 = st.columns([1, 2])
with col2:
    st.session_state.advanced_mode = st.toggle("✨ Enhance with Snowflake Data", value=st.session_state.advanced_mode, 
                                                help="AI corrects domain terminology using your semantic model")

# ---------------------------
# Mic Button
# ---------------------------
st.markdown("""
<style>
    /* Style the audio recorder */
    .stAudioRecorder {
        display: flex;
        justify-content: center;
    }
    
    /* Main mic button */
    .stAudioRecorder > div > button:first-child {
        width: 100px !important;
        height: 100px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #1e3a5f 0%, #29B5E8 100%) !important;
        border: 3px solid rgba(41, 181, 232, 0.3) !important;
        box-shadow: 0 0 30px rgba(41, 181, 232, 0.2), inset 0 0 20px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stAudioRecorder > div > button:first-child:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 40px rgba(41, 181, 232, 0.4), inset 0 0 20px rgba(0,0,0,0.2) !important;
    }
    
    .stAudioRecorder > div > button:first-child svg {
        width: 40px !important;
        height: 40px !important;
    }
    
    /* Hide the save/reset buttons - auto-process instead */
    .stAudioRecorder > div > button:nth-child(2),
    .stAudioRecorder > div > button:nth-child(3),
    .stAudioRecorder > div > div:has(button) button:not(:first-child) {
        display: none !important;
    }
    
    /* Hide any extra controls after recording */
    .stAudioRecorder > div > div {
        display: none !important;
    }
    
    /* Keep only the main button visible */
    .stAudioRecorder > div > button:first-of-type {
        display: flex !important;
    }
</style>
<div class="mic-container">
""", unsafe_allow_html=True)

audio = audiorecorder("", "", show_visualizer=False, key="main_recorder")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Process Audio
# ---------------------------
if len(audio) > 0 and len(audio) != st.session_state.last_audio_len:
    st.session_state.last_audio_len = len(audio)
    st.session_state.pipeline_stage = "processing"
    st.session_state.answer = ""
    
    # Compact audio player
    st.audio(audio.export().read(), format="audio/wav")
    
    try:
        domain_agent = DomainAgent(selected_agent)
    except Exception as e:
        st.error(f"Agent error: {e}")
        st.stop()
    
    # Processing
    with st.spinner(""):
        # Transcribe
        wav_bytes_io = io.BytesIO()
        audio.export(wav_bytes_io, format="wav")
        wav_bytes_io.seek(0)
        with open("temp.wav", "wb") as f:
            f.write(wav_bytes_io.read())
        
        segments, _ = model.transcribe("temp.wav")
        raw_text = " ".join([seg.text for seg in segments]).strip()
        st.session_state.raw_transcript = raw_text
        
        # Enhance if enabled
        if st.session_state.advanced_mode:
            enhanced = domain_agent.correct_transcript(raw_text)
            st.session_state.enhanced_transcript = enhanced
            final_question = enhanced
        else:
            st.session_state.enhanced_transcript = raw_text
            final_question = raw_text
        
        # Get answer
        answer = domain_agent.answer(final_question)
        st.session_state.answer = answer
        st.session_state.pipeline_stage = "complete"

# ---------------------------
# Display Results
# ---------------------------
if st.session_state.pipeline_stage == "complete" and st.session_state.answer:
    
    # Show correction if made
    was_corrected = (st.session_state.advanced_mode and 
                     st.session_state.enhanced_transcript != st.session_state.raw_transcript)
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    st.markdown('<p class="result-label">Your Question</p>', unsafe_allow_html=True)
    
    if was_corrected:
        st.markdown('<span class="correction-badge">✨ Enhanced</span>', unsafe_allow_html=True)
        st.caption(f"~{st.session_state.raw_transcript}~")
    
    st.markdown(f'<p class="result-question">{st.session_state.enhanced_transcript}</p>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="result-label">Answer</p>', unsafe_allow_html=True)
    st.markdown(st.session_state.answer)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Follow-up
    st.markdown("")
    col1, col2 = st.columns([4, 1])
    with col1:
        followup = st.text_input("Follow-up", placeholder="Ask a follow-up...", 
                                  label_visibility="collapsed")
    with col2:
        if st.button("→", use_container_width=True):
            if followup:
                with st.spinner(""):
                    domain_agent = DomainAgent(selected_agent)
                    if st.session_state.advanced_mode:
                        followup = domain_agent.correct_transcript(followup)
                    answer = domain_agent.answer(followup)
                    st.session_state.raw_transcript = followup
                    st.session_state.enhanced_transcript = followup
                    st.session_state.answer = answer
                    st.rerun()
    
    # Reset
    if st.button("New Question", use_container_width=True):
        for key in ["pipeline_stage", "raw_transcript", "enhanced_transcript", "answer", "last_audio_len"]:
            st.session_state[key] = defaults[key]
        st.rerun()
