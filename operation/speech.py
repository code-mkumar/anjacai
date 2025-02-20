import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import streamlit as st
from langdetect import detect
def speak_text(text):
    """Convert text to speech and play it automatically in Streamlit."""
    if not text or not text.strip():  # Check for None or empty string
        print("Error: bot_response is empty or None!")
        return  # Exit function without playing audio
    print(text)
    try:
        language = detect(text)
        
    except Exception as e:
        print(f"Error: {e}")
    tts = gTTS(text=text, lang=language)
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        # Use st.audio to play automatically
        st.audio(temp_file.name, format="audio/mp3", autoplay=True)


def recognize_speech():
    """Capture voice input and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=20, phrase_time_limit=30)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand. Please try again."
        except sr.RequestError:
            return "Speech service is unavailable."
