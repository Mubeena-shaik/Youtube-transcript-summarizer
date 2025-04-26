
import os
import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

import google.generativeai as genai
from googletrans import Translator
from gtts import gTTS
from pydub import AudioSegment
import tempfile

# Streamlit page configuration
st.set_page_config(
    page_title="Gemini Scribe",
    page_icon="📜",
    layout="centered",
)

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
PROMPT = """
Welcome, Video Summarizer! Your task is to distill the essence of a given YouTube video transcript into a concise summary.
Your summary should capture the key points and essential information, presented in bullet points, within a 250-word limit.
"""

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
}

# Utility Functions
@st.cache_data
def extract_transcript(video_id):
    """
    Fetches the transcript for a given YouTube video ID.
    """
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript = " ".join([item["text"] for item in transcript_data])
        return transcript
    except (TranscriptsDisabled, NoTranscriptAvailable, NoTranscriptFound):
        st.warning("Transcript is unavailable for this video. Please try another one.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. Please ensure the video supports captions.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    return None

def split_transcript(transcript, chunk_size=1000):
    """
    Splits a transcript into manageable chunks.
    """
    words = transcript.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_summary(transcript, prompt):
    """
    Generates a summary using the Google Generative AI model (Gemini-Pro).
    """
    try:
        model = genai.GenerativeModel("models/gemma-3-27b-it")  # ✅ Corrected model name
        chunks = split_transcript(transcript)
        summaries = []

        for chunk in chunks:
            response = model.generate_content([prompt, chunk])
            summaries.append(response.text)

        return "\n".join(summaries)

    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def translate_text(text, target_language):
    """
    Translates text into the target language using Google Translate.
    """
    try:
        if not text.strip():
            st.warning("Text is empty. Cannot translate.")
            return None
        translator = Translator()
        return translator.translate(text, dest=target_language).text
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return None

def generate_audio(text, output_file="summary_audio.mp3"):
    """
    Converts text to an audio file using gTTS.
    """
    try:
        if not text.strip():
            raise ValueError("Input text is empty. Cannot generate audio.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts = gTTS(text=text, lang="en")
            tts.save(temp_file.name)
            audio = AudioSegment.from_file(temp_file.name)
            audio.export(output_file, format="mp3")
        return output_file
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Streamlit UI
st.title("🎥 Gemini YouTube Transcript Summarizer")
st.subheader("Extract key insights and generate audio summaries from YouTube videos!")

# User input for YouTube link
youtube_link = st.text_input("Enter YouTube Video Link:")
video_id = None

if youtube_link:
    # Parse video ID
    if "youtu.be" in youtube_link:
        video_id = youtube_link.split("/")[-1]
    elif "youtube.com" in youtube_link:
        if "v=" in youtube_link:
            video_id = youtube_link.split("v=")[1].split("&")[0]

    if video_id:
        # Display thumbnail
        thumbnail_url = f"http://img.youtube.com/vi/{video_id}/0.jpg"
        st.image(thumbnail_url, caption="Video Thumbnail", use_container_width=True)
    else:
        st.error("Could not extract video ID. Please check the link.")

# Language selection
selected_language = st.selectbox("Select the language for detailed notes:", list(LANGUAGES.keys()))

# Generate summary and audio
if st.button("Generate Summary") and video_id:
    transcript = extract_transcript(video_id)
    if transcript:
        # Generate summary
        with st.spinner("Generating summary..."):
            summary = generate_summary(transcript, PROMPT)
        
        if summary:
            # Translate if needed
            target_language = LANGUAGES[selected_language]
            if target_language != "en":
                with st.spinner("Translating summary..."):
                    summary = translate_text(summary, target_language)
            
            # Display summary
            st.markdown(f"### Summary ({selected_language}):")
            st.write(summary)

            # Generate audio file
            with st.spinner("Generating audio..."):
                audio_file = generate_audio(summary, f"summary_audio_{target_language}.mp3")
            
            if audio_file:
                st.audio(audio_file, format="audio/mp3")


