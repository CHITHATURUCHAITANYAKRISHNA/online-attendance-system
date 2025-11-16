import os
import face_recognition

KNOWN_FACES_DIR = "backend/known_faces"


def encode_all_faces():
    print("\n[INFO] Starting face encoding...")

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(KNOWN_FACES_DIR, filename)
            print(f"[ENCODING] Processing {filename}")

            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)

            if len(encodings) == 0:
                print(f"[WARNING] No face found in {filename}. Please use a clear front-facing image.")
            else:
                print(f"[SUCCESS] Face encoded for {filename}")

    print("\n[INFO] Encoding completed!")


if __name__ == "__main__":
    encode_all_faces()
