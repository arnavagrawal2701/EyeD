import cv2
import numpy as np
import sqlite3
import os
from PIL import Image
from CTkMessagebox import CTkMessagebox
from customtkinter import filedialog
import pytz
from datetime import datetime
import pandas as pd

def get_users():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT uName FROM USER")
    users = [row[0] for row in cursor.fetchall()]
    conn.close()
    return users

def insert_or_update_user(uID, uName):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM USER WHERE uID=?", (uID,))
    is_record_exist = cursor.fetchone()

    if is_record_exist:
        cursor.execute("UPDATE USER SET uName=? WHERE uID=?", (uName, uID))
    else:
        cursor.execute("INSERT INTO USER (uID, uName) VALUES (?, ?)", (uID, uName))

    conn.commit()
    conn.close()

def delete_user(uName):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT uID FROM USER WHERE uName=?", (uName,))
    uid = cursor.fetchone()[0]
    # Delete the user by name
    cursor.execute("DELETE FROM USER WHERE uName=?", (uName,))
    conn.commit()
    conn.close()
    return uid

def mark_attendance(uID):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Ensure the attendance table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uID INTEGER,
            uName TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Fetch user details
    cursor.execute("SELECT uName FROM USER WHERE uID=?", (uID,))
    user = cursor.fetchone()
    if user:
        cursor.execute("INSERT INTO Attendance (uID, uName) VALUES (?, ?)", (uID, user[0]))
        conn.commit()
    else:
        CTkMessagebox(title="Error", message="User Not Found!", icon="cancel")
    conn.close()

def collect_data(uID, uName, frame, face_net, sample_num):
    insert_or_update_user(uID, uName)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            sample_num += 1
            face_img = frame[y:y1, x:x1]
            cv2.imwrite(f"dataset/user.{uID}.{sample_num}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            if sample_num >= 40:
                break

    return frame, sample_num  # Return the processed frame and updated sample count

def get_profile(id):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM USER WHERE uID=?", (id,))
    profile = cursor.fetchone()
    conn.close()
    return profile

def detect_faces(frame, face_net, recognizer):
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    h, w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            gray_frame = cv2.cvtColor(frame[y:y1, x:x1], cv2.COLOR_BGR2GRAY)
            id, conf = recognizer.predict(gray_frame)
            if conf < 100:  # Threshold for confidence
                return id  # Return recognized user ID
    return None

def get_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_path:
        face_img = Image.open(single_image_path).convert('L')
        face_np = np.array(face_img, np.uint8)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        faces.append(face_np)
        ids.append(id)

    return np.array(ids), faces

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = "dataset"

    ids, faces = get_images_with_id(path)
    if ids.size == 0:
        os.remove("recognizer/trainingdata.yml")
        return True
    recognizer.train(faces, ids)
    recognizer.save("recognizer/trainingdata.yml")
    cv2.destroyAllWindows()

def get_date():
    """Get the current date in DD-MM-YYYY format for the IST timezone."""
    ist_timezone = pytz.timezone('Asia/Kolkata')
    datetime_utc = datetime.now(pytz.utc)
    datetime_ist = datetime_utc.astimezone(ist_timezone).strftime("%d-%m-%Y_%I-%M-%S")
    return datetime_ist

def exportExcel():
    try:
        # Connect to the database
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()

        # Execute a query to fetch all records from the Students table
        query = "SELECT * FROM Attendance"
        cursor.execute(query)

        # Fetch all rows and get column names
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        # Create a DataFrame from the fetched data
        df = pd.DataFrame(rows, columns=columns)

        # Define the Excel file path
        folder_path = filedialog.askdirectory()
        if folder_path:
            file_path = os.path.join(folder_path, f"student_records_{get_date()}.xlsx")

            # Export DataFrame to Excel
            df.to_excel(file_path, index=False)
            CTkMessagebox(title="File Saved", message=f"Excel file has been saved at {file_path}", icon="check")

        conn.close()
    except Exception as e:
        CTkMessagebox(title="Error", message=f"An error occurred while exporting to Excel: {e}", icon="cancel")
