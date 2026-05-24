# VisionStudio AI

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Flask-WebApp-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/MediaPipe-ComputerVision-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/OpenCV-AI-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/DeepFace-FaceRecognition-red?style=for-the-badge">
</p>

---

# 🚀 VisionStudio AI

VisionStudio AI is a real-time AI-powered computer vision system built using Flask, MediaPipe, OpenCV, TensorFlow, and DeepFace.

The system provides advanced face recognition, hand tracking, gesture interaction, virtual drawing, air writing, and interactive object manipulation directly from a webcam stream.

---

# ✨ Main Features

## 👤 Multi Face Recognition
- Detect multiple faces simultaneously.
- Real-time face recognition.
- Unknown face registration.
- Face database management.
- Rename registered users.
- Delete registered users.
- Add additional face samples for higher accuracy.

---

## ✋ Hand Tracking & Gesture Recognition
- Real-time hand tracking using MediaPipe.
- Finger counting system.
- Finger state detection:
  - Thumb
  - Index
  - Middle
  - Ring
  - Pinky
- Gesture recognition system.
- Custom gesture support.

---

## 🎨 Virtual Drawing & Air Writing
- Draw in the air using hand gestures.
- Air writing interaction.
- Real-time color picker.
- Dynamic brush interaction.
- Smooth hand-controlled drawing system.

---

## 🖱️ Hand Interaction System
- Pinch detection.
- Grab and move text objects.
- Drag & drop interaction.
- Hand-based UI control.
- Interactive virtual workspace.

---

# ⚡ Performance Optimizations

## High Performance Streaming
- Optimized Flask webcam streaming.
- Reduced recognition lag.
- Threaded face recognition processing.
- Independent face tracking system.
- Smooth real-time rendering.

---

# 🧠 AI Technologies Used

| Technology | Purpose |
|---|---|
| Flask | Backend Web Server |
| MediaPipe | Face & Hand Tracking |
| OpenCV | Webcam & Image Processing |
| DeepFace | Face Recognition |
| TensorFlow | AI Inference |
| JavaScript | Frontend Interaction |
| HTML/CSS | User Interface |

---

# 🏗️ Project Architecture

```text
VisionStudio AI
│
├── Flask Backend
│
├── MediaPipe
│   ├── Face Detection
│   ├── Hand Tracking
│   └── Gesture Recognition
│
├── DeepFace
│   └── Face Embedding & Recognition
│
├── OpenCV
│   └── Video Processing
│
└── Frontend
    ├── Live Stream
    ├── Drawing Canvas
    ├── Gesture Interaction
    └── Virtual Workspace
```

---

# 📂 Project Structure

```text
visionstudio-ai/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── face_db.py
│   ├── hand_utils.py
│   └── trackers.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── assets/
│
├── data/
│   ├── known_faces/
│   └── embeddings/
│
└── screenshots/
```

---

# 🔥 Key System Capabilities

✅ Real-time AI Processing  
✅ Multi-face Recognition  
✅ Independent Face Tracking  
✅ Hand Gesture Interaction  
✅ Air Drawing & Writing  
✅ Drag & Drop Using Hand  
✅ Dynamic Object Manipulation  
✅ AI-powered Vision Workspace  
✅ Interactive Human-Computer Interaction  
✅ Optimized Webcam Streaming  

---

# 🖥️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/visionstudio-ai.git
cd visionstudio-ai
```

---

## 2️⃣ Create Virtual Environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Application

```bash
python app.py
```

---

# 🌐 Open In Browser

```text
http://127.0.0.1:5000
```

---

# 📸 Screenshots

> Add screenshots here later.

---

# 🎯 Future Improvements

- AI Sign Language Translation
- Virtual Mouse System
- Gesture-based UI Navigation
- Multi-user Interaction
- 3D Hand Interaction
- AR Object Manipulation
- WebRTC Streaming
- WebSocket Real-time Communication
- AI-powered Virtual Whiteboard
- Voice + Gesture Fusion

---

# 📈 System Goals

VisionStudio AI aims to create a next-generation human-computer interaction system using real-time computer vision and AI technologies.

The project focuses on:
- Natural interaction
- Real-time responsiveness
- AI-powered control systems
- Gesture-based interfaces
- Smart visual interaction

---

# 👨‍💻 Developer

Developed by Tomas Makram

---

# 📄 License

MIT License

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub.
