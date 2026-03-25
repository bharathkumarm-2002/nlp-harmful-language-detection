from PIL import Image
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def predict_image_text(image_path, predict_function):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img).strip()
        if not text:
            return "No text found", "⚠️ Unable to classify"
        result = predict_function(text)
        return text, result
    except Exception as e:
        return f"Error: {str(e)}", "⚠️ Failed"
