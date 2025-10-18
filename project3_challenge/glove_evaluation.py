import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.models import load_model

IMAGE_SIZE = 128
MODEL_PATH = "model.h5"
TEST_DIR = "D:/fs-autonomous-intro-projects/project3_challenge/Dataset/test/combined_test"

def yolo_to_box(bbox, img_width, img_height):
    x_center, y_center, w, h = bbox
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

def load_cropped_data(folder):
    X, y = [], []
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(folder, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(txt_path):
            continue
        height, width = img.shape[:2]
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                x1, y1, x2, y2 = yolo_to_box(bbox, width, height)
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                cropped = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
                cropped = cropped / 255.0
                X.append(cropped)
                y.append(class_id)
    return np.array(X), np.array(y)

print("Loading test dataset...")
X_test, y_test = load_cropped_data(TEST_DIR)
print(f"Loaded {len(X_test)} test samples.")

print("Loading trained model...")
model = load_model(MODEL_PATH)

print("Evaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Loss: {loss:.4f}")

y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Glove", "Glove"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Glove Classification")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Glove", "Glove"]))