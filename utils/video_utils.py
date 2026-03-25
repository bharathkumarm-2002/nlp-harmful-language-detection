import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# Load fine-tuned model (replace path with your own once trained)
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/video_model.pth", map_location=torch.device('cpu')))
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path, fps=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    success, image = cap.read()

    while success:
        if int(cap.get(1)) % (frame_rate * fps) == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image)
            frames.append(pil_img)
        success, image = cap.read()
    cap.release()
    return frames

def classify_video(video_path):
    frames = extract_frames(video_path)
    toxic_scores = []
    
    for frame in frames:
        input_tensor = transform(frame).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.softmax(output, dim=1)
            toxic_scores.append(prediction[0][1].item())  # toxic class score

    avg_score = np.mean(toxic_scores)
    label = "Toxic" if avg_score > 0.5 else "Non-Toxic"
    confidence = round(avg_score * 100, 2) if label == "Toxic" else round((1 - avg_score) * 100, 2)
    return label, confidence
