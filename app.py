
import os
import tempfile
import streamlit as st
from groq import Groq
from gnani.stt import GnaniSTTClient

# Page settings
st.set_page_config(
    page_title="Meeting Notes AI",
    page_icon="📝",
    layout="wide"
)

# Heading
st.title("📝 Meeting Notes AI")
st.subheader("Turn your meeting recordings into smart notes")

st.write(
    "Upload your meeting audio and let AI "
    "generate summaries, key points and action items."
)

# Check API keys
groq_key = os.getenv("GROQ_API_KEY")
gnani_key = os.getenv("GNANI_API_KEY")

if not groq_key or not gnani_key:
    st.error("API keys are missing. Please check your environment variables.")
    st.stop()

# Initialize clients
groq_client = Groq(api_key=groq_key)
gnani_client = GnaniSTTClient(api_key=gnani_key)

# Store transcript between Streamlit reruns
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = ""

st.divider()

# Audio upload
st.header("1. Upload Meeting Audio")

audio_file = st.file_uploader(
    "Choose your meeting recording",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)

st.caption(
    "For this Prisma v3.0 test, use a short recording "
    "of 30 seconds or less."
)

# Transcription
if st.button("🎤 Transcribe Audio", use_container_width=True):

    if audio_file is None:
        st.warning("Please upload an audio file first.")

    else:
        temp_path = None

        try:
            with st.spinner("Gnani AI is converting speech into text..."):

                suffix = os.path.splitext(audio_file.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                ) as temp:
                    temp.write(audio_file.getvalue())
                    temp_path = temp.name

                result = gnani_client.transcribe(
                    temp_path,
                    language_code="indic-auto",
                    model="prisma-v3.0"
                )

                st.session_state.transcript_text = (
                    result.get("transcript", "")
                )

                if not st.session_state.transcript_text.strip():
                    st.warning("No speech was detected.")
                else:
                    st.success("Audio transcribed successfully!")

                    detected = result.get(
                        "detected_language_code",
                        "Not provided"
                    )
                    st.info(f"Detected language: {detected}")

        except Exception as e:
            st.error(f"Gnani transcription error: {e}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

# Display transcript
st.divider()
st.header("2. Meeting Transcript")

transcript = st.text_area(
    "Your meeting conversation",
    key="transcript_text",
    height=220,
    placeholder="Your converted speech will appear here..."
)

# Generate meeting notes
st.divider()
st.header("3. Generate Smart Meeting Notes")

if st.button("✨ Generate Meeting Notes", use_container_width=True):

    if not transcript.strip():
        st.warning("Please upload audio or enter a transcript.")
    else:
        try:
            with st.spinner("Groq AI is preparing your meeting notes..."):

                response = groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are an intelligent meeting assistant.

                            Create professional meeting notes with:

                            1. Meeting Summary
                            2. Key Discussion Points
                            3. Decisions Made
                            4. Action Items and Responsible Persons
                            5. Deadlines
                            6. Next Meeting Details

                            Use clear headings and bullet points.
                            Do not invent missing information.
                            """
                        },
                        {
                            "role": "user",
                            "content": transcript
                        }
                    ]
                )

                notes = response.choices[0].message.content

                st.success("Meeting notes generated!")

                st.markdown(notes)

                st.download_button(
                    label="📥 Download Meeting Notes",
                    data=notes,
                    file_name="meeting_notes.txt",
                    mime="text/plain"
                )

        except Exception as e:
            st.error(f"Groq error: {e}")

st.divider()

st.caption("Powered by Gnani AI + Groq AI")
