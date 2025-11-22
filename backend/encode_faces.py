# backend/encode_faces.py
import os
import cv2
import numpy as np
import json
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.json")

def frame_rgb_from_bgr(frame_bgr):
    import cv2
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def main():
    fa = FaceAnalysis(name='buffalo_l')
    fa.prepare(ctx_id=-1)
    students = []
    for fname in sorted(os.listdir(KNOWN_FACES_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(KNOWN_FACES_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print("Cannot read", path)
            continue
        rgb = frame_rgb_from_bgr(img)
        faces = fa.get(rgb)
        if not faces:
            print("No face:", fname)
            continue
        emb = np.array(faces[0].embedding).tolist()
        reg = os.path.splitext(fname)[0]
        students.append({
            "name": reg,
            "reg_no": reg,
            "dept": "",
            "photo": fname,
            "registered_on": ""
        })
        # optionally save embedding to file if you need
        print("Processed", fname)
    # Save students.json if you want to overwrite (be careful)
    # with open(STUDENTS_FILE, "w", encoding="utf-8") as f:
    #     json.dump(students, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
