from ultralytics import YOLO    
from flask import Flask, request, jsonify
import base64
import os
import shutil
import requests
import gc

app = Flask(__name__)

def delete_runs_folder(folder_path="runs"):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"{folder_path} klasörü silindi.")

def get_latest_predict_folder(base_path="runs/detect"):
    subdirs = [d for d in os.listdir(base_path) if d.startswith("predict")]
    subdirs.sort(key=lambda x: os.path.getctime(os.path.join(base_path, x)), reverse=True)
    return os.path.join(base_path, subdirs[0]) if subdirs else None

def download_image_from_drive(drive_url, save_path="input.jpg"):
    try:
        response = requests.get(drive_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return save_path
        else:
            return None
    except Exception as e:
        print(f"Hata: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    print("İstek alındı!")
    print(request.json)  
    try:
        data = request.get_json()
        if not data or "image_url" not in data:
            return jsonify({"error": "Lütfen 'image_url' parametresi gönderin"}), 400
        
        image_url = data["image_url"]

        delete_runs_folder()

        image_path = download_image_from_drive(image_url)
        if not image_path:
            return jsonify({"error": "Resim indirilemedi"}), 400

        model = YOLO("best.pt")
        model.predict(source=image_path, save=True)

        latest_folder = get_latest_predict_folder()
        if not latest_folder:
            return jsonify({"error": "Predict klasörü bulunamadı"}), 404

        image_files = [f for f in os.listdir(latest_folder) if f.endswith((".jpg", ".png"))]
        if not image_files:
            return jsonify({"error": "Predict klasöründe resim bulunamadı"}), 404
        
        file_path = os.path.join(latest_folder, image_files[0])

        with open(file_path, "rb") as file:
            encoded_string = base64.b64encode(file.read()).decode("utf-8")

        del model
        gc.collect()

        # return jsonify({"file_base64": encoded_string})
        return jsonify({"message": "Test yanıtı"}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
