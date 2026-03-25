import streamlit as st
import os
import tempfile
from PIL import Image, ImageSequence, UnidentifiedImageError
import pytesseract
import speech_recognition as sr
import pandas as pd
# ✅ Add this at the top of app.py
from utils.youtube_utils import fetch_youtube_comments



# ✅ This is the correct import now
from utils import text_utils, audio_utils, gif_utils, image_utils, video_utils, youtube_utils

# ✅ Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Set page config
st.set_page_config(
    page_title="Hate Nirikshak | NLP Harmful Language Classifier",
    page_icon="⚠️",
    layout="wide"
)

# ✅ Load Google Fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ✅ Inject custom CSS (Bootstrap-like styles)
try:
    with open("assets/css/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ Custom style file 'style.css' not found in 'assets/css/'!")

# ✅ Sidebar setup
try:
    st.sidebar.image("images/hate_speech.jpg", width=250)
except:
    st.sidebar.warning("⚠️ 'hate_speech.jpg' not found in 'images/' folder!")

st.sidebar.markdown("## Hate Nirikshak")
nav = st.sidebar.radio("Navigate", ["🏠 Home", "✅ Check Model", "ℹ️ About Us"])

# ✅ Header
st.markdown(
    "<h1 style='text-align: center; color: #0d6efd;'>Hate Speech Detection in English</h1>",
    unsafe_allow_html=True
)

# ... (your imports and setup remain unchanged)

# ==== PAGE LOGIC ====
if nav == "🏠 Home":
    st.subheader("Welcome to Hate Nirikshak")
    st.write("🔍 Detect hate speech in Text, Audio, Video, and more using our multi-modal classifier.")
    st.markdown("""
        ### Key Features:
        - ✅ Text Hate Speech Detection  
        - 🎧 Audio Comment Classification  
        - 🖼️ Image/GIF-based Offensive Detection  
    """)

elif nav == "✅ Check Model":
    st.sidebar.markdown("### Task Selection")
    task = st.sidebar.selectbox("Select Classification Task",
        ["Text Classification", "Audio Classification", 
         "Image Classification", "GIF Classification", 
         "Video Classification","YouTube Comments Classification"])

    # ===== TEXT CLASSIFICATION =====
    if task == "Text Classification":
        st.markdown("### 📝 Text Classification Task")
        user_input = st.text_input("Enter text to classify:", "")
        if st.button("Predict"):
            if user_input:
                result = text_utils.predict_text(user_input)
                if result == "⚠️ Harmful":
                    st.error(f"Prediction: {result}")
                else:
                    st.success(f"Prediction: {result}")
            else:
                st.warning("⚠️ Please enter some text!")

        st.markdown("---")
        st.markdown("#### 🎙️ Or Speak Instead (Mic Access Required)")
        if st.button("🎤 Record & Predict from Voice"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.info("🎙️ Listening... Please speak clearly.")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    st.info("Transcribing...")
                    text = recognizer.recognize_google(audio)
                    st.success(f"🎧 Transcribed Text: {text}")
                    result = text_utils.predict_text(text)
                    st.markdown("#### 🔍 Prediction")
                    if result == "⚠️ Harmful":
                        st.error(f"Prediction: {result}")
                    else:
                        st.success(f"Prediction: {result}")
                except sr.UnknownValueError:
                    st.error("😕 Sorry, couldn't understand the audio.")
                except sr.RequestError:
                    st.error("❌ Could not connect to the speech recognition service.")
                except sr.WaitTimeoutError:
                    st.error("⏱️ Timeout! No speech detected.")

    # ===== AUDIO CLASSIFICATION =====
    elif task == "Audio Classification":
        st.markdown("### 🔊 Audio Classification Task")
        st.write("Upload an audio file (MP3/WAV) to transcribe and classify its content.")

        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

        if uploaded_file:
            st.audio(uploaded_file, format='audio/wav')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                audio_path = tmp_file.name

            if st.button("Transcribe & Predict"):
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_path) as source:
                    st.info("🔁 Transcribing audio to text...")
                    try:
                        audio_data = recognizer.record(source)
                        text = recognizer.recognize_google(audio_data)

                        st.markdown("#### 🎧 Transcribed Text")
                        st.success(text)

                        prediction = text_utils.predict_text(text)
                        st.markdown("#### 🔍 Prediction")
                        if prediction == "⚠️ Harmful":
                            st.error(f"Prediction: {prediction}")
                        else:
                            st.success(f"Prediction: {prediction}")

                    except sr.UnknownValueError:
                        st.warning("😕 Could not understand the audio.")
                    except sr.RequestError:
                        st.error("❌ Could not reach speech recognition service.")

    # ===== IMAGE CLASSIFICATION =====
    elif task == "Image Classification":
        st.markdown("### 🖼️ Image Text Classification Task")
        st.write("Upload an image (JPG/PNG) containing text to classify it.")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                tmp_img.write(uploaded_image.read())
                image_path = tmp_img.name

            if st.button("Analyze Image"):
                extracted_text, prediction = image_utils.predict_image_text(image_path, text_utils.predict_text)
                st.markdown("#### 📝 Extracted Text from Image")
                st.text_area("Extracted Text", extracted_text)
                st.markdown("#### 🔍 Hate Speech Prediction")
                if prediction == "⚠️ Harmful":
                    st.error(f"Prediction: {prediction}")
                else:
                    st.success(f"Prediction: {prediction}")

    # ===== GIF CLASSIFICATION =====
    elif task == "GIF Classification":
        st.markdown("### 🖼️ GIF Text Classification Task")
        st.write("Upload a GIF file (Max 200MB) to extract text and classify.")
        uploaded_gif = st.file_uploader("Upload a GIF", type=["gif"])
        if uploaded_gif:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmp_gif:
                tmp_gif.write(uploaded_gif.read())
                gif_path = tmp_gif.name

            try:
                gif = Image.open(gif_path)
                st.image(gif, caption="Uploaded GIF", use_column_width=True)
                if st.button("Analyze GIF"):
                    extracted_text, prediction = gif_utils.predict_gif_text(gif_path, text_utils.predict_text)
                    st.markdown("#### 📝 Extracted Text from GIF")
                    st.text_area("Extracted Text:", extracted_text)
                    st.markdown("#### 🔍 Hate Speech Prediction")
                    if prediction == "⚠️ Harmful":
                        st.error(f"Prediction: {prediction}")
                    else:
                        st.success(f"Prediction: {prediction}")
            except UnidentifiedImageError:
                st.error("❌ Could not identify the GIF file. Please upload a valid GIF.")

    # ===== VIDEO CLASSIFICATION =====
    elif task == "Video Classification":
        st.markdown("### 📹 Video Classification Task")
        st.write("Upload a video file (MP4/AVI) to classify toxic or non-toxic content.")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])
        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name

            st.video(video_path)
            if st.button("Analyze Video"):
                with st.spinner("Processing video..."):
                    label, confidence = video_utils.classify_video(video_path)
                    st.markdown("#### 🔍 Prediction")
                    if label == "Toxic":
                        st.error(f"Prediction: ⚠️ {label} ({confidence}% confidence)")
                    else:
                        st.success(f"Prediction: ✅ {label} ({confidence}% confidence)")


    elif task == "YouTube Comments Classification":
        st.markdown("### 🎥 YouTube Comments Classification")
        api_key = "AIzaSyAsafcAajsxWi__m01y7ZK2PbLQTWW6E_s"

        query = st.text_input("🔍 Search for a YouTube video:")

        if "youtube_results" not in st.session_state:
            st.session_state.youtube_results = []
        if "selected_index" not in st.session_state:
            st.session_state.selected_index = None

        if query and st.button("Search"):
            with st.spinner("🔎 Searching videos..."):
                results, err = youtube_utils.search_youtube_videos(api_key, query)
                if err:
                    st.error(err)
                elif not results:
                    st.info("No videos found.")
                else:
                    st.session_state.youtube_results = results
                    st.session_state.selected_index = None  # Reset selection

        if st.session_state.youtube_results:
            st.subheader("📺 Select a Video")
            for i, vid in enumerate(st.session_state.youtube_results):
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.image(vid["thumbnail"], width=100)
                with col2:
                    if st.button(f"▶️ {vid['title']}", key=f"vid_{i}"):
                        st.session_state.selected_index = i

        # After selection
        if st.session_state.selected_index is not None:
            selected_video = st.session_state.youtube_results[st.session_state.selected_index]
            selected_video_url = f"https://www.youtube.com/watch?v={selected_video['video_id']}"
            st.markdown(f"🎬 **Selected Video:** {selected_video_url}")
            st.video(selected_video_url)

            if st.button("📊 Classify Comments"):
                with st.spinner("Fetching comments and classifying..."):
                    df, err = youtube_utils.fetch_youtube_comments(api_key, selected_video_url)
                    if err:
                        st.error(err)
                    elif df.empty:
                        st.warning("No comments found or video may be restricted.")
                    else:
                        st.success("✅ Classification Complete")
                        st.dataframe(df)

        # Reset button
        if st.button("🔄 Reset"):
            st.session_state.selected_index = None
            st.session_state.youtube_results = []



elif nav == "ℹ️ About Us":
    st.subheader("👨‍💻 About Us")
    st.write("""
        This project is built by your team for identifying and classifying **harmful language** using NLP and multimodal learning.
        
        It is designed to detect toxic, hate, and offensive content from:
        - 📝 Text
        - 🎧 Audio
        - 🖼️ Images & GIFs

        Technologies Used:
        - Python, Streamlit, Scikit-learn, Pandas
        - Pre-trained ML models with custom preprocessing
    """)

        # Visit: Google Cloud Console - Credentials