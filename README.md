# 🎓 Online Attendance System (Face Recognition)

A face-recognition–based attendance system built using **Python**, **Flask**, **OpenCV**, and **face_recognition**.  
The system automatically detects, recognizes, and marks attendance of students using their facial features.

---

## 🚀 Features

- 📸 Real-time face recognition  
- 🎯 High accuracy using `face_recognition` + `dlib`  
- 📝 Automatic attendance logging in JSON files  
- 👨‍🏫 Admin login & dashboard  
- 👩‍🎓 Student registration with face encoding  
- 🌐 Simple frontend (HTML) + Flask backend  
- 🖼 Stores known faces in the `backend/known_faces/` folder  
- 📁 Fully reorganized clean folder structure  

---

## 📂 Project Structure

attendance-face/
│── backend/
│ ├── app.py
│ ├── encode_faces.py
│ ├── models.py
│ ├── admin.json
│ ├── students.json
│ ├── attendance.json
│ ├── known_faces/
│ ├── requirements.txt
│ └── Procfile (for Render deployment)
│
└── frontend/
├── index.html
├── admin.html
├── admin_login.html
├── analytics.html
└── register.html

yaml
Copy code

---

## 🛠 Tech Stack

### 🔹 Backend
- Python 3
- Flask
- Flask-CORS
- OpenCV (`opencv-python`)
- face_recognition (uses dlib)
- NumPy  

### 🔹 Frontend
- HTML, CSS, JavaScript

---

## 📦 Installation (Local Setup)

### 1️⃣ Clone the repository
```bash
git clone https://github.com/CHITHATURUCHAITANYAKRISHNA/online-attendance-system.git
cd online-attendance-system/backend
2️⃣ Create virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install dependencies
⚠️ dlib requires CMake. Make sure it is installed before running.

bash
Copy code
pip install -r requirements.txt
4️⃣ Run backend
bash
Copy code
python app.py
5️⃣ Open frontend
Open the frontend/index.html file in your browser.

🌐 Deployment (Important)
⚠️ dlib is NOT supported on:
Render

Railway

Vercel

Netlify

Koyeb

These platforms cannot build dlib.

✔️ Supported on:
Local machine

Google Colab (recommended)

Custom VPS (Linux server)

Windows Server

▶️ Run Online Using Google Colab (Recommended)
Google Colab supports dlib + GPU.

You can use:

ngrok

localtunnel

cloudflared

To expose your backend publicly.

📊 JSON Data Files
File	Description
students.json	Registered students with face encodings
attendance.json	Daily attendance records
admin.json	Admin login credentials

👤 Author
CHITHATURUCHAITANYAKRISHNA
🔗 GitHub: https://github.com/CHITHATURUCHAITANYAKRISHNA

⭐ If you like this project
Please give a star ⭐ on the repository. It motivates further improvements!
