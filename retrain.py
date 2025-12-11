import cv2
import os
import numpy as np

# Path to your images
TRAINING_DIR = "TrainingImages"

# Haarcascade file
HAAR_FILE = "haarcascade_frontalface_default.xml"

# Load Haarcascade
face_detector = cv2.CascadeClassifier(HAAR_FILE)

# Create LBPH Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
name_list = []
label_id = 0

print("------ Retraining Model ------")

# Loop through Student & Staff folders
for role in ["Student", "Staff"]:
    role_path = os.path.join(TRAINING_DIR, role)

    if not os.path.isdir(role_path):
        continue

    # Loop each person's folder
    for folder in os.listdir(role_path):
        folder_path = os.path.join(role_path, folder)

        if not os.path.isdir(folder_path):
            continue

        label_id += 1
        name_list.append(folder)

        print(f"Training: {folder}")

        # Loop all images inside folder
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)

            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(label_id)

# Convert to NumPy arrays
faces = np.array(faces)
labels = np.array(labels)

# Train model
recognizer.train(faces, labels)

# Save trained model
recognizer.save("trainer.yml