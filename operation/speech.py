import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import streamlit as st
from langdetect import detect
import torchaudio
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
    """Capture voice input using torchaudio and convert to text."""
    recognizer = sr.Recognizer()

    # Use a pre-recorded file instead of Microphone (Streamlit Cloud limitation)
    audio_file = "recorded_audio.wav"

    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Save it as a WAV file for recognition
        torchaudio.save("temp_audio.wav", waveform, sample_rate)

        # Use speech_recognition to transcribe the saved file
        with sr.AudioFile("temp_audio.wav") as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)

        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand. Please try again."
    except sr.RequestError:
        return "Speech service is unavailable."
    except Exception as e:
        return f"Error: {e}"
