# backend/app.py  — full corrected version (matches analytics frontend format)
import os
import json
import io
import csv
import base64
import threading
from datetime import datetime
from typing import Tuple, List, Dict

from flask import (
    Flask, request, jsonify, send_from_directory, session, send_file, redirect
)
from flask_cors import CORS
from werkzeug.utils import secure_filename

import face_recognition
import cv2
import numpy as np

# -------------------------
# Paths & config
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../frontend"))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.json")
STUDENTS_FILE = os.path.join(BASE_DIR, "students.json")
ADMIN_FILE = os.path.join(BASE_DIR, "admin.json")

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Ensure JSON files exist
for path, default in [
    (ATTENDANCE_FILE, []),
    (STUDENTS_FILE, []),
    (ADMIN_FILE, [{"username": "admin", "password": "admin"}])
]:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)
app.secret_key = "replace_this_with_a_random_secret_key"

# Thread-safety
attendance_lock = threading.Lock()
students_lock = threading.Lock()
known_lock = threading.Lock()

# In-memory face database
known_encodings: List[np.ndarray] = []
known_names: List[str] = []   # stores reg_no (preferred) or filename key fallback

# -------------------------
# JSON helpers
# -------------------------
def read_json(path: str):
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -------------------------
# Utilities
# -------------------------
def clean_name_from_filename(filename: str) -> str:
    """Strip repeated image extensions and path, preserve case."""
    if not filename:
        return ""
    name = os.path.basename(filename)
    while True:
        base, ext = os.path.splitext(name)
        if ext.lower() in (".jpg", ".jpeg", ".png"):
            name = base
            continue
        break
    return name

def load_known_faces():
    """Load face encodings from KNOWN_FACES_DIR into memory and map to reg_no if possible."""
    global known_encodings, known_names
    with known_lock:
        known_encodings = []
        known_names = []

        students = read_json(STUDENTS_FILE)
        photo_to_reg = {}
        for s in students:
            photo = s.get("photo")
            reg = s.get("reg_no")
            if photo and reg:
                photo_to_reg[clean_name_from_filename(photo)] = reg

        for file in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            key = clean_name_from_filename(file)
            img_path = os.path.join(KNOWN_FACES_DIR, file)
            try:
                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if not encs:
                    app.logger.warning(f"[load_known_faces] skipping {file}: no encodings")
                    continue
                known_encodings.append(encs[0])
                known_names.append(photo_to_reg.get(key, key))
            except Exception as e:
                app.logger.warning(f"[load_known_faces] failed {file}: {e}")

# initial load
load_known_faces()

# -------------------------
# Admin protection & pages
# -------------------------
@app.before_request
def protect_admin_pages():
    # protect SPA admin pages (URLs served by static files)
    protected = ["/admin", "/register", "/analytics"]
    # allow static assets, api, etc.
    if request.path in protected:
        if not session.get("admin_logged_in"):
            return redirect("/admin_login_page")
    return None

@app.route("/admin_login_page")
def admin_login_page():
    # serve the admin login static file (frontend should contain admin_login.html)
    return app.send_static_file("admin_login.html")

@app.route("/admin_login", methods=["POST"])
def admin_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")
    admins = read_json(ADMIN_FILE)
    for a in admins:
        if a.get("username") == username and a.get("password") == password:
            session["admin_logged_in"] = True
            session["admin_user"] = username
            return jsonify({"ok": True})
    return jsonify({"ok": False, "message": "Invalid credentials"}), 401

@app.route("/admin_logout", methods=["POST"])
def admin_logout():
    session.clear()
    return jsonify({"ok": True})

@app.route("/is_logged_in", methods=["GET"])
def is_logged_in():
    return jsonify({"logged_in": bool(session.get("admin_logged_in", False))})

# -------------------------
# Image decode helper
# -------------------------
def decode_base64_image(data_url: str) -> Tuple[bool, np.ndarray or str]:
    """Return (ok, frame or error_message)"""
    if not data_url or "," not in data_url:
        return False, "Invalid image data"
    try:
        header, b64 = data_url.split(",", 1)
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return False, "Failed to decode image"
        return True, frame
    except Exception as e:
        return False, f"Decode error: {e}"

# -------------------------
# Mark attendance
# -------------------------
@app.route("/mark_attendance", methods=["POST"])
def mark_attendance():
    try:
        payload = request.get_json(silent=True) or {}
        image_data = payload.get("image")
        if not image_data:
            return jsonify({"status": "error", "message": "No image"}), 400

        ok, frame_or_err = decode_base64_image(image_data)
        if not ok:
            return jsonify({"status": "error", "message": frame_or_err}), 400
        frame = frame_or_err

        # resize (keep faces large enough)
        h, w = frame.shape[:2]
        scale = 0.6 if max(h, w) < 800 else 0.5
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # detect faces (HOG for compatibility)
        try:
            locations = face_recognition.face_locations(rgb_small, model="hog")
        except Exception as e:
            app.logger.warning(f"face_locations error: {e}")
            return jsonify({"status": "error", "message": "Face detection error"}), 500

        if not locations:
            return jsonify({"status": "no_face", "message": "No face detected"})

        encodings = face_recognition.face_encodings(rgb_small, locations)
        if not encodings:
            return jsonify({"status": "error", "message": "Encoding failed"})

        with attendance_lock:
            attendance = read_json(ATTENDANCE_FILE)

        today = datetime.now().strftime("%Y-%m-%d")

        # compare each found face
        for enc in encodings:
            with known_lock:
                if not known_encodings:
                    continue
                matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
                distances = face_recognition.face_distance(known_encodings, enc)
            best_idx = int(np.argmin(distances)) if len(distances) else None

            if best_idx is not None and matches[best_idx]:
                reg_no = known_names[best_idx]

                students = read_json(STUDENTS_FILE)
                display_name = next((s.get("name") for s in students if s.get("reg_no") == reg_no), None)
                if not display_name:
                    # try photo key match
                    display_name = next((s.get("name") for s in students
                                         if clean_name_from_filename(s.get("photo", "")) == str(reg_no)), None)
                if not display_name:
                    display_name = clean_name_from_filename(str(reg_no))

                # Prevent multiple marks same day
                if any(a.get("reg_no") == reg_no and a.get("date") == today for a in attendance):
                    return jsonify({"status": "exists", "message": "Attendance already marked today",
                                    "name": display_name, "reg_no": reg_no})

                # Add entry
                now = datetime.now()
                entry = {
                    "name": display_name,
                    "reg_no": reg_no,
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S")
                }

                with attendance_lock:
                    attendance = read_json(ATTENDANCE_FILE)
                    attendance.append(entry)
                    write_json(ATTENDANCE_FILE, attendance)

                return jsonify({"status": "success", "name": display_name, "reg_no": reg_no})

        return jsonify({"status": "unknown", "message": "Face not recognized"})
    except Exception as e:
        app.logger.exception("mark_attendance error")
        return jsonify({"status": "error", "error": str(e)}), 500

# -------------------------
# Register student (upload file OR blob from canvas)
# -------------------------
@app.route("/register_student", methods=["POST"])
def register_student():
    try:
        name = request.form.get("name")
        reg_no = request.form.get("reg_no")
        dept = request.form.get("dept")
        file = request.files.get("photo")

        if not (name and reg_no and dept and file):
            return jsonify({"message": "All fields required"}), 400

        # sanitize & ensure extension
        _, ext = os.path.splitext(secure_filename(file.filename or ""))
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        filename = f"{reg_no}{ext.lower()}"
        path = os.path.join(KNOWN_FACES_DIR, filename)
        file.save(path)

        # verify face in uploaded image
        try:
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)
        except Exception:
            enc = []

        if not enc:
            if os.path.exists(path):
                os.remove(path)
            return jsonify({"message": "No face detected in uploaded photo"}), 400

        # append to in-memory lists
        with known_lock:
            known_encodings.append(enc[0])
            known_names.append(reg_no)

        # add to students.json
        with students_lock:
            students = read_json(STUDENTS_FILE)
            students.append({
                "name": name,
                "reg_no": reg_no,
                "dept": dept,
                "photo": filename,
                "registered_on": datetime.utcnow().isoformat()
            })
            write_json(STUDENTS_FILE, students)

        return jsonify({"ok": True, "message": f"{name} registered"})
    except Exception as e:
        app.logger.exception("register_student error")
        return jsonify({"message": str(e)}), 500

# -------------------------
# Analytics endpoint (format 'A' requested)
# Returns: total_students, total_attendance, attendance_by_date, top_students, attendance_by_dept, recent
# -------------------------
@app.route("/analytics_data", methods=["GET"])
def analytics_data():
    attendance = read_json(ATTENDANCE_FILE)
    students = read_json(STUDENTS_FILE)

    total_students = len(students)
    total_attendance = len(attendance)

    # attendance_by_date: {date: count}
    by_date: Dict[str, int] = {}
    # by_student: {reg_no: count}
    by_student: Dict[str, int] = {}
    # by_dept: {dept: count}
    reg_to_dept = {s.get("reg_no"): s.get("dept", "Unknown") for s in students}
    by_dept: Dict[str, int] = {}

    for a in attendance:
        date = a.get("date")
        reg = a.get("reg_no")
        if date:
            by_date[date] = by_date.get(date, 0) + 1
        if reg:
            by_student[reg] = by_student.get(reg, 0) + 1
            dept = reg_to_dept.get(reg, "Unknown")
            by_dept[dept] = by_dept.get(dept, 0) + 1

    # convert to required arrays
    attendance_by_date = sorted([[d, c] for d, c in by_date.items()])
    top_students = sorted([[r, c] for r, c in by_student.items()], key=lambda x: x[1], reverse=True)[:10]
    attendance_by_dept = sorted([[d, c] for d, c in by_dept.items()])

    recent = list(reversed(attendance))[:50]

    return jsonify({
        "total_students": total_students,
        "total_attendance": total_attendance,
        "attendance_by_date": attendance_by_date,
        "top_students": top_students,
        "attendance_by_dept": attendance_by_dept,
        "recent": recent
    })

# -------------------------
# Download CSV
# -------------------------
@app.route("/download_attendance", methods=["GET"])
def download_attendance():
    attendance = read_json(ATTENDANCE_FILE)
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(["Name", "Reg No", "Date", "Time"])
    for a in attendance:
        writer.writerow([a.get("name", ""), a.get("reg_no", ""), a.get("date", ""), a.get("time", "")])
    mem = io.BytesIO(si.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="attendance.csv")

# -------------------------
# Get students & attendance APIs
# -------------------------
@app.route("/get_students", methods=["GET"])
def get_students():
    return jsonify(read_json(STUDENTS_FILE))

@app.route("/get_attendance", methods=["GET"])
def get_attendance():
    return jsonify(read_json(ATTENDANCE_FILE))

# -------------------------
# Delete student
# -------------------------
@app.route("/delete_student", methods=["POST"])
def delete_student():
    if not session.get("admin_logged_in"):
        return jsonify({"ok": False, "message": "login required"}), 401

    data = request.get_json(silent=True) or {}
    reg_no = data.get("reg_no")
    if not reg_no:
        return jsonify({"ok": False, "message": "reg_no required"}), 400

    # remove from students.json
    with students_lock:
        students = read_json(STUDENTS_FILE)
        students = [s for s in students if s.get("reg_no") != reg_no]
        write_json(STUDENTS_FILE, students)

    # remove image file (any extension)
    for ext in (".jpg", ".jpeg", ".png"):
        path = os.path.join(KNOWN_FACES_DIR, f"{reg_no}{ext}")
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

    # remove from in-memory lists
    with known_lock:
        for i in range(len(known_names) - 1, -1, -1):
            if known_names[i] == reg_no:
                known_names.pop(i)
                known_encodings.pop(i)

    return jsonify({"ok": True})

# -------------------------
# Serve student photos + static pages
# -------------------------
@app.route("/student_photo/<path:filename>")
def student_photo(filename):
    return send_from_directory(KNOWN_FACES_DIR, filename)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/admin")
def admin_page():
    return app.send_static_file("admin.html")

@app.route("/register")
def register_page():
    return app.send_static_file("register.html")

@app.route("/analytics")
def analytics_page():
    return app.send_static_file("analytics.html")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # Use use_reloader=False to avoid duplicate processes when reloading
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
