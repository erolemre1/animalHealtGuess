from ultralytics import YOLO    
from flask import Flask, request, jsonify
import base64
import os
import shutil

def delete_runs_folder(folder_path="runs"):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} klasörü silindi.")

model = YOLO("best.pt")

delete_runs_folder()
model.predict(source="25.jpg", save=True)

app = Flask(__name__)

def get_latest_predict_folder(base_path="runs/detect"):
    """En son oluşturulan predict klasörünü bulur."""
    subdirs = [d for d in os.listdir(base_path) if d.startswith("predict")]
    subdirs.sort(key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
    return os.path.join(base_path, subdirs[0]) if subdirs else None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        latest_folder = get_latest_predict_folder()
        if not latest_folder:
            return jsonify({"error": "Predict klasörü bulunamadı"}), 404
        
        image_files = [f for f in os.listdir(latest_folder) if f.endswith((".jpg", ".png"))]
        if not image_files:
            return jsonify({"error": "Predict klasöründe resim bulunamadı"}), 404
        
        file_path = os.path.join(latest_folder, image_files[0])

        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("utf-8")

        return jsonify({"file_base64": encoded_string})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
