import cv2
import numpy as np
import os
from PIL import Image
import customtkinter as ctk
import util
from CTkMessagebox import CTkMessagebox

def convert_frame_to_ctkimage(frame, size):
    """Convert OpenCV BGR frame to CTkImage for CustomTkinter."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = Image.fromarray(rgb_frame)  # Convert to PIL Image
    ctk_image = ctk.CTkImage(light_image=img, size=size)  # Convert to CTkImage
    return ctk_image

# Helper function to convert OpenCV image to PIL Image
class AttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("800x500")

        # Face detection model
        self.face_net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists("recognizer/trainingdata.yml"):
            self.recognizer.read("recognizer/trainingdata.yml")

        self.cam = cv2.VideoCapture(0)
        self.user_options = util.get_users()

        # Layout Frames
        self.left_frame = ctk.CTkFrame(self, width=400, height=400)
        self.left_frame.pack(side="left", padx=10, pady=10)
        self.right_frame = ctk.CTkFrame(self, width=400, height=400)
        self.right_frame.pack(side="right", padx=10, pady=10)

        # Camera Display
        self.camera_label = ctk.CTkLabel(self.left_frame, text="")
        self.camera_label.pack(padx=5, pady=5)

        # Right Frame Widgets
        self.status_label = ctk.CTkLabel(self.right_frame, text="Look at the Camera", font=("Helvetica", 14))
        self.status_label.pack(pady=10)

        self.mark_btn = ctk.CTkButton(self.right_frame, text="Mark Present", command=self.mark_attendance)
        self.mark_btn.pack(pady=5)

        # Manager Frame
        self.manager_label = ctk.CTkLabel(self.right_frame, text="Password:")
        self.manager_entry = ctk.CTkEntry(self.right_frame)
        self.manager_login_btn = ctk.CTkButton(self.right_frame, text="Manager Login", command=self.manager_login)
        self.manager_login_btn.pack(pady=5)

        self.mng_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.user_dropdown = ctk.CTkOptionMenu(self.mng_frame, values=self.user_options)
        self.delete_btn = ctk.CTkButton(self.mng_frame, text="Delete User", command=self.delete_user)
        self.register_btn = ctk.CTkButton(self.mng_frame, text="Register", command=self.show_register_form)
        self.export_btn = ctk.CTkButton(self.mng_frame, text="Export", command=util.exportExcel)
        self.change_pass_btn = ctk.CTkButton(self.mng_frame, text="Change Password", command=self.change_password)
        self.back_btn = ctk.CTkButton(self.mng_frame, text="Back", command=self.home_screen)

        self.mng_password = "1234"
        self.mng_newpass_entry = ctk.CTkEntry(self.mng_frame, placeholder_text="New Password")
        self.mng_confirm_newpass_entry = ctk.CTkEntry(self.mng_frame, placeholder_text="Confirm New Password")

        self.close_btn = ctk.CTkButton(self.right_frame, text="Close", command=self.close)
        self.close_btn.pack(pady=5)

        # Registration Form
        self.reg_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.reg_title = ctk.CTkLabel(self.reg_frame, text="Registration Form", font=("Helvetica", 14))
        self.reg_title.pack()

        self.id_entry = ctk.CTkEntry(self.reg_frame, placeholder_text="Employee ID")
        self.name_entry = ctk.CTkEntry(self.reg_frame, placeholder_text="Employee Name")
        self.submit_btn = ctk.CTkButton(self.reg_frame, text="Submit", command=self.register_user)
        self.form_forget = ctk.CTkButton(self.reg_frame, text= "Close Form", command=self.close_form)

        self.update_camera()

    def home_screen(self):
        self.mng_frame.pack_forget()
        self.user_dropdown.pack_forget()
        self.delete_btn.pack_forget()
        self.register_btn.pack_forget()
        self.change_pass_btn.pack_forget()
        self.export_btn.pack_forget()
        self.back_btn.pack_forget()
    def manager_login(self):
        if self.manager_entry.get():
            if self.manager_entry.get() == "1234":
                self.mng_frame.pack(pady=10)
                self.user_dropdown.pack(pady=5)
                self.delete_btn.pack(pady=5)
                self.register_btn.pack(pady=5)
                self.change_pass_btn.pack(pady=5)
                self.export_btn.pack(pady=5)
                self.back_btn.pack(pady=5)
                self.manager_label.pack_forget()
                self.manager_entry.pack_forget()
            else:
                CTkMessagebox(title="Error", message="Incorrect Password, Try again", icon="cancel")

        else:
            self.manager_label.pack(pady=2)
            self.manager_entry.pack(pady=2)



    def update_camera(self):
        ret, frame = self.cam.read()
        if ret:
            blob = cv2.dnn.blobFromImage(
                frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
            )
            self.face_net.setInput(blob)
            detections = self.face_net.forward()

            h, w = frame.shape[:2]
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            img_ctk = convert_frame_to_ctkimage(frame, (400, 300))
            self.camera_label.configure(image=img_ctk)
            self.camera_label.image = img_ctk
        self.after(10, self.update_camera)

    def mark_attendance(self):
        ret, frame = self.cam.read()
        if ret:
            profile_id = util.detect_faces(frame, self.face_net, self.recognizer)
            if profile_id:
                util.mark_attendance(profile_id)
                profile = util.get_profile(profile_id)
                if profile:
                    CTkMessagebox(title="Success", message=f"Attendance Marked for {profile[1]}", icon="check")
            else:
                CTkMessagebox(title="Error", message="Face not recognized. Try again.", icon="cancel")

    def delete_user(self):
        selected_user = self.user_dropdown.get()
        if selected_user and selected_user != "Delete User":
            msg = CTkMessagebox(title="Confirm Delete", message=f"Are you sure you want to delete user {selected_user}?",
                                icon="warning", option_1="No", option_2="Yes")
            confirm = msg.get()
            if confirm == "Yes":
                util.delete_user(selected_user)
                util.train_recognizer()
                self.user_dropdown.configure(values=util.get_users())
                CTkMessagebox(title="Success", message=f"User {selected_user} had been deleted.", icon="check")
        else:
            CTkMessagebox(title="Error", message="Select a valid user to delete.")

    def close_form(self):
        self.submit_btn.pack_forget()
        self.reg_frame.pack_forget()
        self.form_forget.pack_forget()

    def show_register_form(self):
        self.reg_frame.pack(pady=10)
        self.id_entry.pack(pady=5)
        self.name_entry.pack(pady=5)
        self.submit_btn.pack(pady=5)
        self.form_forget.pack()

    def register_user(self):
        user_id = self.id_entry.get()
        user_name = self.name_entry.get()
        if user_id and user_name:
            sample_num = 0
            while sample_num < 40:
                ret, frame = self.cam.read()
                if ret:
                    processed_frame, sample_num = util.collect_data(
                        user_id, user_name, frame, self.face_net, sample_num
                    )
                    img_ctk = convert_frame_to_ctkimage(processed_frame, (400, 300))
                    self.camera_label.configure(image=img_ctk)
                    self.camera_label.image = img_ctk
            util.train_recognizer()
            CTkMessagebox(title="Success", message="Registration Successful")
            self.close_form()

    def change_password(self):
        if self.mng_newpass_entry.get():
            if self.mng_confirm_newpass_entry.get():
                if self.mng_newpass_entry.get() == self.mng_confirm_newpass_entry.get():
                    self.mng_password = self.mng_confirm_newpass_entry.get()
                else:
                    CTkMessagebox(title="Error", message="Password Change Unsuccessful!", icon="cancel")
        else:
            self.mng_newpass_entry.pack(pady=5)
            self.mng_confirm_newpass_entry.pack(pady=5)



    def close(self):
        self.destroy()

if __name__ == "__main__":
    if not os.path.exists("dataset"):
        os.makedirs("dataset")
    if not os.path.exists("recognizer"):
        os.makedirs("recognizer")

    app = AttendanceApp()
    app.mainloop()

