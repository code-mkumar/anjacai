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

import torchaudio.backend.sox_io_backend
import soundfile as sf
# Ensure the correct backend is used
torchaudio.set_audio_backend("soundfile")  # Use "sox_io" if supported

def recognize_speech(audio_file):
    """Transcribe audio from file using SpeechRecognition."""
    recognizer = sr.Recognizer()

    try:
        # Load audio using soundfile (alternative to torchaudio)
        waveform, sample_rate = sf.read(audio_file)

        # Save it as a WAV file for SpeechRecognition
        sf.write("temp_audio.wav", waveform, sample_rate)

        # Use SpeechRecognition
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
