import io
import os
import yaml
import streamlit as st
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel
from cerebras.cloud.sdk import Cerebras
from typing import Optional


# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="Local Whisper + Cerebras", page_icon="🎙️")
st.title("🎙️ Local Whisper Transcriber with Cerebras + YAML Context")

st.write(
    "Record audio below and transcribe it locally using Whisper (faster-whisper). "
    "Then clean up the transcript using a Cerebras LLM, enriched with Snowflake-specific YAML context."
)

# ---------------------------
# Cached resources
# ---------------------------
@st.cache_resource
def load_whisper_model():
    # choose: tiny, base, small, medium, large
    return WhisperModel("small", device="cpu")


@st.cache_resource
def get_cerebras_client():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CEREBRAS_API_KEY environment variable is not set. "
            "Please set it before running the app."
        )
    return Cerebras(api_key=api_key)


@st.cache_resource
def load_yaml_context():
    """
    Load Snowflake-specific semantic context from YAML.
    Expects a file named 'snowflake_specific_terms.yaml'
    in the same directory as this app.
    """
    yaml_path = "snowflake_specific_terms.yaml"
    if not os.path.exists(yaml_path):
        st.warning(f"YAML semantic context file not found: {yaml_path}")
        return None

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        st.error("❌ Failed to parse snowflake_specific_terms.yaml")
        st.exception(e)
        return None


model = load_whisper_model()
yaml_context = load_yaml_context()

# Lazy init of Cerebras client only if we actually need it
cerebras_client = None

# ---------------------------
# Helper: refine transcript with Cerebras
# ---------------------------
def refine_transcript_with_cerebras(
    raw_text: str, semantic_views: Optional[dict] = None
) -> str:
    """
    Send the raw Whisper transcript + optional semantic views
    (like YAML context, SQL definitions, etc.) to a Cerebras LLM
    and get back cleaned / corrected text.
    Handles errors gracefully.
    """
    global cerebras_client
    if cerebras_client is None:
        cerebras_client = get_cerebras_client()

    # Turn semantic views into a readable string for the model
    blocks = []
    if semantic_views:
        for key, value in semantic_views.items():
            if not value:
                continue

            # If structured (dict/list), pretty-dump to YAML
            if isinstance(value, (dict, list)):
                pretty = yaml.dump(value, sort_keys=False)
                blocks.append(f"{key} (YAML):\n```yaml\n{pretty}\n```")
            else:
                blocks.append(f"{key}: {value}")

    views_str = "\n\n".join(blocks) if blocks else "(none provided)"

    system_prompt = (
        "You are a transcription post-processor.\n"
        "You receive raw automatic speech recognition (ASR) text and optional semantic context.\n\n"
        "Your tasks:\n"
        "- Fix spelling, grammar, and capitalization.\n"
        "- Add punctuation and sentence boundaries.\n"
        "- Use the provided semantic context (YAML, domain terms, schemas, etc.) "
        "to interpret and standardize Snowflake-specific terminology.\n"
        "- Preserve technical names and Snowflake-specific terminology as given.\n"
        "- Do NOT invent new entities that are not present in the semantic context "
        "unless clearly implied.\n\n"
        "Return only the cleaned text. No explanations."
    )

    user_content = f"""Raw transcript:
\"\"\"{raw_text}\"\"\"

Semantic context:
{views_str}
"""

    try:
        response = cerebras_client.chat.completions.create(
            model="llama-3.3-70b",  # change model if needed
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )

        # Validate the response
        if (
            not hasattr(response, "choices")
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            st.error("⚠️ Cerebras returned an empty or invalid response.")
            return ""

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error("❌ Cerebras LLM failed to process the request.")
        st.exception(e)  # prints full traceback in Streamlit
        return ""


# ---------------------------
# Session state for transcript
# ---------------------------
if "raw_transcript" not in st.session_state:
    st.session_state.raw_transcript = ""

# ---------------------------
# Audio recorder
# ---------------------------
audio = audiorecorder("🔴 Click to start / stop recording", "⏺️ Recording...")

if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")
    st.success("Recording captured! Click the button below to transcribe.")

    if st.button("📝 Transcribe locally with Whisper"):
        with st.spinner("Transcribing... this may take a few seconds"):
            # Export audio to WAV bytes
            wav_bytes_io = io.BytesIO()
            audio.export(wav_bytes_io, format="wav")
            wav_bytes_io.seek(0)

            # Save to a temp file (Whisper needs a path)
            with open("temp.wav", "wb") as f:
                f.write(wav_bytes_io.read())

            # Run Whisper locally
            segments, info = model.transcribe("temp.wav")

            # Collect transcript text
            transcript_text = " ".join([seg.text for seg in segments])

        st.session_state.raw_transcript = transcript_text

# ---------------------------
# Show raw transcript (if any)
# ---------------------------
if st.session_state.raw_transcript:
    st.subheader("Raw transcript (Whisper)")
    st.write(st.session_state.raw_transcript)

    # Optionally show loaded YAML for your own visibility
    if yaml_context is not None:
        with st.expander("View loaded Snowflake YAML semantic context"):
            st.json(yaml_context)

    # ---------------------------
    # Cerebras cleanup button
    # ---------------------------
    if st.button("✨ Clean & correct with Cerebras (using YAML context)"):
        # Build semantic views dict, include YAML if present
        semantic_views = {}
        if yaml_context is not None:
            semantic_views["snowflake_specific_terms"] = yaml_context

        with st.spinner("Refining transcript with Cerebras..."):
            refined_text = refine_transcript_with_cerebras(
                st.session_state.raw_transcript,
                semantic_views=semantic_views,
            )

        if refined_text:
            st.success("✅ Cerebras successfully processed your request.")
            st.subheader("Refined transcript (Cerebras)")
            st.write(refined_text)
