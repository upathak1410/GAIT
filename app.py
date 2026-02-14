from fastapi import FastAPI, UploadFile, File
import shutil
import os
from inference import predict_video

app = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {"message": "Gait Xception Silhouette API Running"}

@app.post("/predict-video")
async def predict_video_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded video
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_video(file_path)

    # Delete video after processing
    os.remove(file_path)

    if prediction is None:
        return {"error": "No frames processed"}

    return {"predicted_subject_id": prediction}
