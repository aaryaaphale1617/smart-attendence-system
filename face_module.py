# face_module.py
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

CASCADE_PATH = "haarcascade_frontalface_default.xml"
TRAINER_FILE = "trainer.yml"
NAMES_FILE = "names.txt"
TRAINING_DIR = "TrainingImages"
ATT_FILE = "data/attendance.csv"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# create recognizer object (will be trained later)
def load_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists(TRAINER_FILE):
        recognizer.read(TRAINER_FILE)
    return recognizer

def get_names_list():
    if os.path.exists(NAMES_FILE):
        return open(NAMES_FILE).read().splitlines()
    return []

def mark_attendance(name, role="Student"):
    os.makedirs(os.path.dirname(ATT_FILE), exist_ok=True)
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    row = [name, role, date, time]
    df = pd.DataFrame([row])
    header = not os.path.exists(ATT_FILE) or os.path.getsize(ATT_FILE) == 0
    df.to_csv(ATT_FILE, mode='a', header=header, index=False)

def detect_and_recognize_frame(frame, recognizer, names):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    results = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            id_, conf = recognizer.predict(roi)
        except:
            id_, conf = -1, 999
        name = "Unknown"
        if conf < 70 and 0 <= id_ < len(names):
            name = names[id_]
        results.append({
            "box": (x, y, w, h),
            "name": name,
            "conf": conf
        })
    return results

def start_recognition_session(mark_all=True, duration_sec=15):
    """
    Start camera, run recognition for duration_sec seconds or until 'q' key pressed.
    mark_all -> if True, mark attendance for both students and staff when detected.
    Returns last_frame (BGR), list of detected names during session.
    """
    recognizer = load_recognizer()
    names = get_names_list()
    cap = cv2.VideoCapture(0)
    start = datetime.now()
    detected = set()
    last_frame = None

    while (datetime.now() - start).seconds < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        results = detect_and_recognize_frame(frame, recognizer, names)
        for r in results:
            x, y, w, h = r["box"]
            n = r["name"]
            conf = r["conf"]
            color = (0,255,0) if n != "Unknown" else (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"{n} {round(conf,1)}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            if n != "Unknown" and n not in detected:
                # Determine role from students.csv and staff.csv
                role = "Student"
                try:
                    s = pd.read_csv("data/students.csv")
                    if n in s["Name"].astype(str).values:
                        role = "Student"
                    else:
                        stf = pd.read_csv("data/staff.csv")
                        if n in stf["Name"].astype(str).values:
                            role = "Staff"
                except:
                    role = "Unknown"
                if mark_all:
                    mark_attendance(n, role)
                detected.add(n)
        # show last frame if needed by UI
        # break if needed by UI
    cap.release()
    return last_frame, list(detected)

# TRAINING helper: read TrainingImages folder and create trainer.yml
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    ids = []
    names = []
    label_map = {}  # name -> id
    current_id = 0

    if not os.path.exists(TRAINING_DIR):
        return False, "No training images"

    for person_name in sorted(os.listdir(TRAINING_DIR)):
        person_path = os.path.join(TRAINING_DIR, person_name)
        if not os.path.isdir(person_path):
            continue
        if person_name in label_map:
            idx = label_map[person_name]
        else:
            label_map[person_name] = current_id
            idx = current_id
            current_id += 1
            names.append(person_name)
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                continue
            faces.append(img_arr)
            ids.append(idx)

    if len(faces) == 0:
        return False, "No valid face images found."

    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)

    # write names in order of label id
    ordered_names = [None] * len(label_map)
    for name, idx in label_map.items():
        ordered_names[idx] = name
    with open(NAMES_FILE, "w") as f:
        for n in ordered_names:
            f.write(str(n) + "\n")

    return True, f"Trained {len(ordered_names)} persons"