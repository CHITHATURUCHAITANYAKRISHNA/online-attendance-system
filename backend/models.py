import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENTS_FILE = os.path.join(BASE_DIR, "students.json")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.json")
ADMIN_FILE = os.path.join(BASE_DIR, "admin.json")


# ---------------------------------------------------------
# Helper: Load JSON file
# ---------------------------------------------------------
def load_json(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)

    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


# ---------------------------------------------------------
# Helper: Save JSON file
# ---------------------------------------------------------
def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------
# STUDENT FUNCTIONS
# ---------------------------------------------------------

def get_all_students():
    return load_json(STUDENTS_FILE)


def add_student(name, reg_no, dept, photo_filename):
    students = load_json(STUDENTS_FILE)

    new_student = {
        "name": name,
        "reg_no": reg_no,
        "dept": dept,
        "photo": photo_filename,
        "registered_on": datetime.now().isoformat()
    }

    students.append(new_student)
    save_json(STUDENTS_FILE, students)
    return True


def delete_student(reg_no):
    students = load_json(STUDENTS_FILE)
    updated = [s for s in students if s["reg_no"] != reg_no]
    save_json(STUDENTS_FILE, updated)
    return len(students) != len(updated)


def find_student_by_photo(filename):
    students = load_json(STUDENTS_FILE)
    for s in students:
        if s["photo"] == filename:
            return s
    return None


def find_student_by_reg(reg_no):
    students = load_json(STUDENTS_FILE)
    for s in students:
        if s["reg_no"] == reg_no:
            return s
    return None


# ---------------------------------------------------------
# ATTENDANCE FUNCTIONS
# ---------------------------------------------------------

def add_attendance(photo_filename):
    attendance = load_json(ATTENDANCE_FILE)

    stu = find_student_by_photo(photo_filename)

    record = {
        "name": stu["name"] if stu else photo_filename,
        "reg_no": stu["reg_no"] if stu else "unknown",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S")
    }

    attendance.append(record)
    save_json(ATTENDANCE_FILE, attendance)
    return record


def get_all_attendance():
    return load_json(ATTENDANCE_FILE)


def reset_attendance():
    save_json(ATTENDANCE_FILE, [])
    return True


# ---------------------------------------------------------
# ADMIN FUNCTIONS
# ---------------------------------------------------------

def load_admin():
    if not os.path.exists(ADMIN_FILE):
        with open(ADMIN_FILE, "w") as f:
            json.dump({"username": "admin", "password": "admin"}, f)

    with open(ADMIN_FILE, "r") as f:
        return json.load(f)


def verify_admin(username, password):
    admin = load_admin()
    return username == admin["username"] and password == admin["password"]
