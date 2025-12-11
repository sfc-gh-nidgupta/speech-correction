import io
import os
import re
import time
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from agent_manager import get_agent_manager, DomainAgent

# ---------------------------
# Page Config & Custom Styling
# ---------------------------
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 50%, #0d1117 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #29b5e8 0%, #7dd3fc 50%, #29b5e8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(41, 181, 232, 0.15);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #7dd3fc;
        margin-bottom: 1rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTextArea textarea {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: white;
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
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "ready"
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""
if "enhanced_transcript" not in st.session_state:
    st.session_state.enhanced_transcript = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "last_audio_len" not in st.session_state:
    st.session_state.last_audio_len = 0
if "selected_agent" not in st.session_state:
    st.session_state.selected_agent = None
if "show_create_agent" not in st.session_state:
    st.session_state.show_create_agent = False

# ---------------------------
# Sidebar - Agent Selection & Creation
# ---------------------------
with st.sidebar:
    st.markdown("### 🤖 Agent Selection")
    
    # Get available agents
    agents = agent_manager.list_agents()
    
    if agents:
        agent_options = {f"{a.icon} {a.name}": a for a in agents}
        selected_name = st.selectbox(
            "Choose an agent:",
            options=list(agent_options.keys()),
            index=0
        )
        selected_agent_config = agent_options[selected_name]
        st.session_state.selected_agent = selected_agent_config
        
        # Show agent info
        st.caption(selected_agent_config.description)
    else:
        st.warning("No agents found. Create one below!")
        selected_agent_config = None
    
    st.markdown("---")
    
    # Create Agent Section
    if st.button("➕ Create New Agent", use_container_width=True):
        st.session_state.show_create_agent = not st.session_state.show_create_agent
    
    if st.session_state.show_create_agent:
        st.markdown("### 📝 Create Agent")
        
        new_agent_name = st.text_input("Agent Name", placeholder="e.g., My Domain")
        new_agent_icon = st.text_input("Icon (emoji)", value="🤖", max_chars=2)
        new_agent_desc = st.text_input("Description", placeholder="Expert for...")
        
        new_agent_terms = st.text_area(
            "Terms YAML",
            height=150,
            placeholder="""domain_terms:
  - Term1
  - Term2
acronyms:
  - ABC (Full Name)""",
            help="YAML format: list of domain-specific terms for speech correction"
        )
        
        new_agent_knowledge = st.text_area(
            "Knowledge Base (Markdown)",
            height=150,
            placeholder="""# FAQ

Q: What is X?
A: X is...""",
            help="Markdown FAQ that the agent uses to answer questions"
        )
        
        if st.button("✅ Create Agent", type="primary", use_container_width=True):
            if new_agent_name and new_agent_terms and new_agent_knowledge:
                # Create agent ID from name
                agent_id = re.sub(r'[^a-z0-9]+', '_', new_agent_name.lower()).strip('_')
                
                try:
                    agent_manager.create_agent(
                        agent_id=agent_id,
                        name=new_agent_name,
                        description=new_agent_desc or f"Expert for {new_agent_name}",
                        icon=new_agent_icon,
                        terms_content=new_agent_terms,
                        knowledge_content=new_agent_knowledge
                    )
                    st.success(f"Created agent: {new_agent_icon} {new_agent_name}")
                    st.session_state.show_create_agent = False
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating agent: {e}")
            else:
                st.warning("Please fill in all required fields")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    advanced_mode = st.toggle("✨ Query Enhancement", value=True,
                               help="AI enhances your question with proper terminology")
    
    st.markdown("---")
    
    # Pipeline status
    st.markdown("### 📊 Pipeline")
    stages = [
        ("🎙️", "Record", st.session_state.pipeline_stage in ["transcribing", "enhancing", "answering", "complete"]),
        ("📝", "Transcribe", st.session_state.pipeline_stage in ["enhancing", "answering", "complete"]),
        ("✨", "Enhance", st.session_state.pipeline_stage in ["answering", "complete"] and advanced_mode),
        ("💬", "Answer", st.session_state.pipeline_stage == "complete"),
    ]
    for icon, label, complete in stages:
        if label == "Enhance" and not advanced_mode:
            continue
        status = "✅" if complete else "○"
        st.markdown(f"{status} {icon} {label}")

# ---------------------------
# Main Content
# ---------------------------
if selected_agent_config:
    # Header with agent info
    st.markdown(f'<h1 class="main-header">{selected_agent_config.icon} {selected_agent_config.name} Voice Assistant</h1>', 
                unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{selected_agent_config.description}</p>', unsafe_allow_html=True)
    
    # Recording section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f'<div class="agent-badge">🎤 Voice-Powered • AI-Enhanced</div>', unsafe_allow_html=True)
        
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
        
        # Initialize domain agent
        try:
            domain_agent = DomainAgent(selected_agent_config)
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            st.stop()
        
        with st.status("🎯 Processing your question...", expanded=True) as status:
            # Step 1: Transcribe
            st.write("📝 **Transcribing audio...**")
            
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
                st.write("✨ **Enhancing with domain terminology...**")
                
                enhanced = domain_agent.correct_transcript(raw_text)
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
            st.write(f"{selected_agent_config.icon} **Consulting knowledge base...**")
            
            answer = domain_agent.answer(final_question)
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
        
        # Follow-up
        st.markdown("---")
        st.markdown("### 🔄 Follow-up")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            followup = st.text_input("Type a follow-up question:",
                                      placeholder="e.g., Tell me more about...",
                                      label_visibility="collapsed")
        with col2:
            if st.button("Ask →", type="primary", use_container_width=True):
                if followup:
                    with st.spinner("Getting answer..."):
                        domain_agent = DomainAgent(selected_agent_config)
                        if advanced_mode:
                            followup = domain_agent.correct_transcript(followup)
                        answer = domain_agent.answer(followup)
                        st.session_state.raw_transcript = followup
                        st.session_state.enhanced_transcript = followup
                        st.session_state.answer = answer
                        st.rerun()
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Ask Another Question", use_container_width=True):
            st.session_state.pipeline_stage = "ready"
            st.session_state.raw_transcript = ""
            st.session_state.enhanced_transcript = ""
            st.session_state.answer = ""
            st.session_state.last_audio_len = 0
            st.rerun()

else:
    st.markdown('<h1 class="main-header">🎙️ Voice Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Create an agent to get started</p>', unsafe_allow_html=True)
    
    st.info("👈 Click **Create New Agent** in the sidebar to create your first agent with a terms YAML and knowledge base.")
