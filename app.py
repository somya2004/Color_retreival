import os
import pickle
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from PIL import Image

FEATURES_FILE = "features.npy"
IMAGE_PATHS_FILE = "image_paths.pkl"

features = np.load(FEATURES_FILE)
with open(IMAGE_PATHS_FILE, "rb") as f:
    image_paths = pickle.load(f)

app = Flask(__name__)

def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    hist = image.histogram()
    hist = np.array(hist) / np.sum(hist)
    return hist

def compute_similarity(query_feature, dataset_features):
    distances = np.linalg.norm(dataset_features - query_feature, axis=1)
    return distances

@app.route('/dataset/<path:filename>')
def dataset_files(filename):
    return send_from_directory('dataset', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file:
            img = Image.open(file.stream)
            query_feature = extract_features(img)
            distances = compute_similarity(query_feature, features)

            top_k = 6
            idx = np.argsort(distances)[:top_k]

            # Convert absolute paths → URL paths Flask can serve
            retrieved_images = []
            for p in [image_paths[i] for i in idx]:
                filename = os.path.basename(p)
                retrieved_images.append(f"/dataset/{filename}")

            return render_template("index.html", results=retrieved_images)

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
