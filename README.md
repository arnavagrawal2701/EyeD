# EyeD - Face Recognition Attendance System

## Overview

This project implements a **Face Recognition-based Attendance System** with a graphical user interface built using **CustomTkinter**.

It uses:

- A **DNN-based face detector** (Caffe model: `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
- An **LBPH face recogniser** (`cv2.face.LBPHFaceRecognizer_create`) for identifying registered users.
- A **SQLite database (`attendance.db`)** to store user details and attendance.
- An **Excel export** feature for attendance records.

This project is aligned with **Unit 5 – Applications of Computer Vision** of the Computer Vision syllabus.

## Features

- Register new users with employee ID and name.
- Capture face samples using the webcam and store them in the `dataset/` folder.
- Train an LBPH face recogniser and store the model in `recognizer/trainingdata.yml`.
- Mark attendance automatically when a recognised user looks at the camera.
- Store attendance records in a SQLite database (`attendance.db`).
- Export attendance records to Excel (`.xlsx`) using a GUI folder chooser.
- Manager login to:
  - Register users
  - Delete users
  - Export attendance
  - Change manager password (in-memory for this version)

## Technologies Used

- Python
- OpenCV (DNN for face detection, LBPH for face recognition)
- NumPy
- Pillow
- CustomTkinter
- CTkMessagebox
- SQLite (`sqlite3`)
- Pandas + OpenPyXL
- pytz (for IST timestamps)

## Project Structure

Suggested structure:

```text
face_attendance/
├─ main.py                        # AttendanceApp (GUI + logic)
├─ util.py                        # Helper functions (DB, training, export)
├─ deploy.prototxt                # Face detection model definition
├─ res10_300x300_ssd_iter_140000.caffemodel   # Face detection model weights
├─ attendance.db                         # SQLite database (USER, Attendance)
├─ dataset/                       # Captured face images
│   └─ user.<uID>.<sample>.jpg
├─ recognizer/
│   └─ trainingdata.yml           # Trained LBPH recogniser model
├─ requirements.txt
└─ README.md
