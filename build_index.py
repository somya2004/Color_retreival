import os
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image

dataset_folder = "dataset/"

model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feat = model.predict(x)
    return feat.flatten()

features = []
paths = []

files = os.listdir(dataset_folder)

for file in files:
    path = os.path.join(dataset_folder, file)
    if path.lower().endswith((".jpg", ".png", ".jpeg")):
        print("Processing:", path)
        paths.append(path)
        features.append(extract_features(path))

features = np.array(features)

np.save("features.npy", features)

with open("image_paths.pkl", "wb") as f:
    pickle.dump(paths, f)

print("Index built successfully!")
