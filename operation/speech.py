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
        

# import torchaudio.backend.sox_io_backend
# import sounddevice as sd
# import numpy as np
# import scipy.io.wavfile as wav


# def record_and_transcribe(duration=5, sample_rate=44100):
#     """Records audio from the microphone, processes it with torchaudio, and transcribes it."""
#     st.info(f"Recording for {duration} seconds... Speak now!")

#     # Step 1: Record Audio
#     audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
#     sd.wait()  # Wait for recording to finish
#     wav.write("recorded_audio.wav", sample_rate, audio_data)
#     st.success("Recording complete!")

#     # Step 2: Load with Torchaudio
#     recognizer = sr.Recognizer()
#     try:
#         waveform, sample_rate = torchaudio.load("recorded_audio.wav")

#         # Save the processed file for SpeechRecognition
#         torchaudio.save("temp_audio.wav", waveform, sample_rate)

#         # Step 3: Transcribe using SpeechRecognition
#         with sr.AudioFile("temp_audio.wav") as source:
#             audio = recognizer.record(source)
#             text = recognizer.recognize_google(audio)

#         return text
#     except sr.UnknownValueError:
#         return "Sorry, I couldn't understand. Please try again."
#     except sr.RequestError:
#         return "Speech service is unavailable."
#     except Exception as e:
#         return f"Error: {e}"
