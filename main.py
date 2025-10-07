# main.py

import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# 1. Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="YOLOv8 Dedicated Detection API",
    description="API dengan endpoint terpisah untuk deteksi kuku dan lidah.",
    version="3.0.0"
)

# 2. Muat kedua model YOLOv8 saat aplikasi dijalankan
try:
    model_kuku = YOLO("models/kuku_best.pt")
    model_lidah = YOLO("models/lidah_best.pt")
except Exception as e:
    print(f"Peringatan: Gagal memuat satu atau lebih model. Error: {e}")
    model_kuku = None
    model_lidah = None

# 3. Endpoint utama untuk menyapa pengguna
@app.get("/")
def read_root():
    """Endpoint dasar untuk mengecek apakah API berjalan."""
    return {"message": "Welcome to the YOLO Dedicated Detection API!"}

# Fungsi bantuan untuk menemukan deteksi terbaik (menghindari duplikasi kode)
def get_best_detection_from_result(result, class_name_target, threshold):
    """Memproses hasil prediksi dan mengembalikan deteksi terbaik."""
    best_detection = None
    for box in result.boxes:
        confidence = float(box.conf)
        class_name = result.names[int(box.cls)]
        
        if class_name == class_name_target and confidence > threshold:
            if best_detection is None or confidence > best_detection["confidence"]:
                best_detection = {
                    "class_name": class_name,
                    "confidence": confidence,
                    "box_normalized": box.xyxyn.tolist()[0]
                }
    return best_detection

# 4. Endpoint KHUSUS untuk deteksi KUKU
@app.post("/detect/kuku")
async def detect_kuku(file: UploadFile = File(...)):
    """Endpoint untuk mendeteksi 'kuku' dari sebuah gambar."""
    if not model_kuku:
        raise HTTPException(status_code=500, detail="Model untuk 'kuku' tidak tersedia.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="File gambar tidak valid.")

    CONFIDENCE_THRESHOLD = 0.5
    results = model_kuku.predict(image, verbose=False)
    best_detection = get_best_detection_from_result(results[0], "kuku", CONFIDENCE_THRESHOLD)

    if best_detection:
        return {
            "detection": best_detection,
            "message": "Kuku terlihat"
        }
    else:
        return {
            "detection": None,
            "message": "Kuku tidak terlihat"
        }

# 5. Endpoint KHUSUS untuk deteksi LIDAH
@app.post("/detect/lidah")
async def detect_lidah(file: UploadFile = File(...)):
    """Endpoint untuk mendeteksi 'lidah' dari sebuah gambar."""
    if not model_lidah:
        raise HTTPException(status_code=500, detail="Model untuk 'lidah' tidak tersedia.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="File gambar tidak valid.")

    CONFIDENCE_THRESHOLD = 0.5
    results = model_lidah.predict(image, verbose=False)
    best_detection = get_best_detection_from_result(results[0], "lidah", CONFIDENCE_THRESHOLD)

    if best_detection:
        return {
            "detection": best_detection,
            "message": "Lidah terlihat"
        }
    else:
        return {
            "detection": None,
            "message": "Lidah tidak terlihat"
        }
