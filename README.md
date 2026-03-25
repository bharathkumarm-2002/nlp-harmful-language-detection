# NLP-Based Harmful Language Detection and Classification

## 📌 Project Overview
This project is a multi-modal NLP system that detects and classifies harmful or toxic language from different types of inputs such as text, audio, images, videos, and YouTube comments.

The system converts all inputs into text or frames and uses machine learning models to classify whether the content is harmful or non-harmful.

---

## 🚀 Features
- 🔤 Text-based harmful language detection  
- 🎤 Audio to text classification  
- 🖼 Image OCR-based detection  
- 🎞 GIF frame analysis  
- 🎥 Video frame classification  
- 📺 YouTube comment filtering  
- 🖥 User-friendly interface using Streamlit  

---

## 🧠 How the Project Works

### 1. Text Input
- User enters text  
- Text is preprocessed  
- Passed to ML model  
- Output: Harmful / Not Harmful  

### 2. Audio Input
- Audio is converted to text using Speech Recognition  
- Text is classified using NLP model  

### 3. Image Input
- Text extracted using OCR (Tesseract)  
- Classified using NLP model  

### 4. GIF Input
- Frames extracted from GIF  
- Text extracted and classified  

### 5. Video Input
- Frames extracted from video  
- Each frame analyzed using model  

### 6. YouTube Comments
- Comments fetched using API  
- Each comment classified  

---

## 🛠 Technologies Used
- Python  
- Streamlit (Frontend UI)  
- Machine Learning (Scikit-learn, PyTorch)  
- NLP (Text Classification)  
- OpenCV (Video Processing)  
- Tesseract OCR (Image Text Extraction)  
- Speech Recognition (Audio Processing)  

---

## 📂 Project Structure
hate_speech_bot3/
│── app.py
│── preprocess.py
│── train_model.py
│── train_video_model.py
│── extract_video_frames.py
│
├── datasets/
├── models/
├── utils/
├── templates/
├── static/
├── video_data/


## ⚙️ Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/bharathkumarm-2002/nlp-harmful-language-detection.git
cd nlp-harmful-language-detection

**Step 2: Create Virtual Environment (Optional)**

```bash
python -m venv venv
venv\Scripts\activate

Step 3: Install Dependencies 
```bash
pip install -r requirements.txt

Step 4: Install Tesseract OCR
Download from: https://github.com/tesseract-ocr/tesseract
After installation, update path in code if needed.

▶️ Run the Project
```bash
streamlit run app.py

Open in browser:
http://localhost:8501
