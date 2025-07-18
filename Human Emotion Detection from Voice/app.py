# app.py
import streamlit as st
import sounddevice as sd
import wavio
import joblib
import numpy as np
from feature_extraction import extract_features
from scipy.io.wavfile import write

st.set_page_config(page_title="Voice Emotion Detector")
st.title("🎙️ Human Emotion Detection from Voice")
st.write("Record your voice and detect the underlying emotion.")

model = joblib.load("emotion_model.pkl")

duration = 5  # in seconds
fs = 44100

# Initialize session state
if "history" not in st.session_state:
    st.session_state["history"] = []
if "recorded" not in st.session_state:
    st.session_state["recorded"] = False
    
# Button to record
if st.button("🎤 Record Voice"):
    try:
        st.info("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        wavio.write("temp.wav", recording, fs, sampwidth=2)
        st.success("✅ Recording complete.")
        st.audio("temp.wav", format='audio/wav')
    except Exception as e:
        st.error(f"Microphone error: {e}")
        st.warning("🎙️ Please check if your microphone is connected, enabled, and accessible to Python.")


    features = extract_features("temp.wav").reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"🔊 Detected Emotion: **{prediction.upper()}**")

    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append(prediction)

    st.write("📊 Emotion Trend")
    emotion_list = list(set(st.session_state["history"]))
    freq = [st.session_state["history"].count(e) for e in emotion_list]
    st.bar_chart(data=dict(zip(emotion_list, freq)))

    # Save to log file
    with open("session_log.txt", "a") as f:
        f.write(prediction + "\n")


# # app.py
# import streamlit as st
# import sounddevice as sd
# import wavio
# import joblib
# import numpy as np
# import os
# from feature_extraction import extract_features

# # Page config
# st.set_page_config(page_title="Voice Emotion Detector")
# st.title("🎙️ Human Emotion Detection from Voice")
# st.write("Record your voice and detect the underlying emotion.")

# # Load trained model
# model = joblib.load("emotion_model.pkl")

# # Constants
# duration = 5  # seconds
# fs = 44100
# filename = "temp.wav"

# # Initialize session state
# if "history" not in st.session_state:
#     st.session_state["history"] = []
# if "recorded" not in st.session_state:
#     st.session_state["recorded"] = False

# # 🎤 RECORD BUTTON
# if st.button("🎤 Record Voice"):
#     try:
#         st.info("Recording for 5 seconds...")
#         recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#         sd.wait()
#         wavio.write(filename, recording, fs, sampwidth=2)
#         st.session_state["recorded"] = True
#         st.success("✅ Recording saved as temp.wav")
#         st.audio(filename, format='audio/wav')
#     except Exception as e:
#         st.error(f"Microphone Error: {e}")
#         st.session_state["recorded"] = False

# # 🔍 PREDICT BUTTON
# if st.session_state["recorded"] and st.button("🔍 Detect Emotion"):
#     if os.path.exists(filename):
#         try:
#             features = extract_features(filename).reshape(1, -1)
#             prediction = model.predict(features)[0]
#             st.success(f"🔊 Detected Emotion: **{prediction.upper()}**")

#             # Save history
#             st.session_state["history"].append(prediction)

#             # Show chart
#             st.write("📊 Emotion Trend")
#             emotion_list = list(set(st.session_state["history"]))
#             freq = [st.session_state["history"].count(e) for e in emotion_list]
#             st.bar_chart(data=dict(zip(emotion_list, freq)))

#             # Save to log
#             with open("session_log.txt", "a") as f:
#                 f.write(prediction + "\n")
#         except Exception as e:
#             st.error(f"Feature extraction error: {e}")
#     else:
#         st.error("❌ temp.wav not found. Please record your voice first.")

# # ℹ️ Tip
# if not st.session_state["recorded"]:
#     st.info("Please click '🎤 Record Voice' before trying to detect emotion.")
