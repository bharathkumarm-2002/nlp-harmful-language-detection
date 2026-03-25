from PIL import Image, ImageSequence
import pytesseract

# ✅ DO NOT ADD QUOTES
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_gif(gif_path):
    gif = Image.open(gif_path)
    all_text = []

    for frame in ImageSequence.Iterator(gif):
        frame = frame.convert("RGB")
        text = pytesseract.image_to_string(frame)
        if text.strip():
            all_text.append(text.strip())

    return " ".join(all_text)

def predict_gif_text(gif_path, text_predict_function):
    extracted_text = extract_text_from_gif(gif_path)
    if not extracted_text.strip():
        return "No text found in GIF", "Not Harmful"

    prediction = text_predict_function(extracted_text)
    return extracted_text, prediction
