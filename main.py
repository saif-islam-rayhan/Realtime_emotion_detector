import cv2
import numpy as np
import os
import uuid
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from typing import List
import base64
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware







# ===============================
# App Init
# ===============================
app = FastAPI(title="Emotion Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # অথবা frontend এর URL দিন
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)





BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ===============================
# Load Model
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("emotion_model.h5 not found")

model = load_model(MODEL_PATH)

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

emotion_colors = {
    "Angry": (0, 0, 255),      # Red
    "Disgust": (0, 128, 0),    # Green
    "Fear": (255, 0, 255),     # Purple
    "Happy": (0, 255, 255),    # Yellow
    "Sad": (255, 0, 0),        # Blue
    "Surprise": (255, 165, 0), # Orange
    "Neutral": (255, 255, 255) # White
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# Helpers
# ===============================
def prepare_face(face_gray):
    try:
        face = cv2.resize(face_gray, (48, 48))
        face = face.astype("float32") / 255.0
        return face.reshape(1, 48, 48, 1)
    except:
        return None

def predict_emotion(face_gray):
    x = prepare_face(face_gray)
    if x is None:
        return None, None
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    return emotion_labels[idx], float(preds[idx])

def detect_all_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    results = []

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        label, conf = predict_emotion(face_gray)

        if label:
            color = emotion_colors.get(label, (0, 255, 0))
            
            results.append({
                "emotion": label,
                "confidence": round(conf * 100, 2),
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Draw emotion label with background
            label_text = f"{label} ({conf*100:.1f}%)"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(frame, 
                         (x, y - text_height - 10), 
                         (x + text_width, y), 
                         color, -1)
            
            # Put text
            cv2.putText(
                frame,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0) if label != "Neutral" else (0, 0, 0),
                2
            )

    return frame, results

# ===============================
# Frontend Routes
# ===============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ===============================
# IMAGE API
# ===============================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}.jpg"
    input_path = os.path.join(UPLOAD_DIR, filename)
    output_path = os.path.join(PROCESSED_DIR, filename)

    # Read and save original image
    contents = await file.read()
    with open(input_path, "wb") as f:
        f.write(contents)

    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Process image
    processed, predictions = detect_all_emotions(image)
    cv2.imwrite(output_path, processed)

    # Convert processed image to base64 for immediate display
    _, buffer = cv2.imencode('.jpg', processed)
    img_str = base64.b64encode(buffer).decode()

    return {
        "faces_detected": len(predictions),
        "predictions": predictions,
        "output_image": f"/processed/{filename}",
        "image_base64": f"data:image/jpeg;base64,{img_str}"
    }

# ===============================
# VIDEO API
# ===============================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    # Generate unique filename
    ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    input_path = os.path.join(UPLOAD_DIR, unique_filename)
    output_path = os.path.join(PROCESSED_DIR, unique_filename)

    # Save uploaded video
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Process video
    cap = cv2.VideoCapture(input_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = 0
    faces_per_frame = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, predictions = detect_all_emotions(frame)
        out.write(processed_frame)
        
        faces_per_frame.append(len(predictions))
        total_frames += 1
    
    cap.release()
    out.release()
    
    # Generate thumbnail
    cap = cv2.VideoCapture(output_path)
    ret, thumbnail_frame = cap.read()
    if ret:
        thumbnail_path = output_path.replace(ext, '_thumb.jpg')
        cv2.imwrite(thumbnail_path, thumbnail_frame)
    
    cap.release()
    
    return {
        "message": "Video processed successfully",
        "output_video": f"/processed/{unique_filename}",
        "thumbnail": f"/processed/{unique_filename.replace(ext, '_thumb.jpg')}",
        "total_frames": total_frames,
        "average_faces": sum(faces_per_frame) / max(len(faces_per_frame), 1)
    }

# ===============================
# WEBCAM API
# ===============================
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process frame for emotion detection
        processed_frame, _ = detect_all_emotions(frame)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.get("/video-feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# ===============================
# SERVE PROCESSED FILES
# ===============================
@app.get("/processed/{filename}")
async def get_processed_file(filename: str):
    file_path = os.path.join(PROCESSED_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)

# ===============================
# HEALTH CHECK
# ===============================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}