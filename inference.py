import torch
import cv2
import numpy as np
from collections import Counter
from torchvision import transforms
from PIL import Image

MODEL_PATH = "gait_xception_model_final123_subjects.pth"

device = torch.device("cpu")
model = timm.create_model('xception', pretrained=False, num_classes=123)

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.eval()

# EXACT SAME TRANSFORM AS TRAINING
transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
])

def convert_to_silhouette(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Convert back to 3-channel
    silhouette = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    return silhouette

def preprocess_frame(frame):
    silhouette = convert_to_silhouette(frame)
    image = Image.fromarray(silhouette)
    image = transform(image).unsqueeze(0)
    return image

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    predictions = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 5th frame for speed
        if frame_count % 5 == 0:
            input_tensor = preprocess_frame(frame)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                predictions.append(pred)

        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        return None

    final_prediction = Counter(predictions).most_common(1)[0][0]
    return final_prediction
