import tkinter as tk
from tkinter import messagebox, font
import cv2
import os
import numpy as np
from PIL import Image
import threading
import datetime

DATASET_DIR = "dataset"
LOG_FILE = "log.txt"
MODEL_FILE = "lbph_model.yml"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

def log_action(action, name="Unknown"):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.datetime.now()} - {action}: {name}\n")

class FacialAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Facial Authentication - LBPH")
        self.root.geometry("450x300")
        self.root.configure(bg="#2c3e50")

        # Fonts
        self.title_font = font.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = font.Font(family="Helvetica", size=12)
        self.button_font = font.Font(family="Helvetica", size=12, weight="bold")

        # Title Label
        self.title_label = tk.Label(root, text="Facial Authentication System", bg="#2c3e50", fg="white", font=self.title_font)
        self.title_label.pack(pady=15)

        # Name Entry Label
        self.name_label = tk.Label(root, text="Enter Name (for registration):", bg="#2c3e50", fg="white", font=self.label_font)
        self.name_label.pack(pady=5)

        # Name Entry
        self.entry_name = tk.Entry(root, font=self.label_font)
        self.entry_name.pack(pady=5)

        # Register Button
        self.register_button = tk.Button(root, text="Register", command=self.start_register_thread, bg="#27ae60", fg="white", font=self.button_font, activebackground="#2ecc71", activeforeground="white")
        self.register_button.pack(pady=10, ipadx=10, ipady=5)

        # Login Button
        self.login_button = tk.Button(root, text="Login", command=self.start_login_thread, bg="#2980b9", fg="white", font=self.button_font, activebackground="#3498db", activeforeground="white")
        self.login_button.pack(pady=10, ipadx=10, ipady=5)

        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        if os.path.exists(MODEL_FILE):
            self.recognizer.read(MODEL_FILE)

    def start_register_thread(self):
        thread = threading.Thread(target=self.register)
        thread.daemon = True
        thread.start()

    def start_login_thread(self):
        thread = threading.Thread(target=self.login_user)
        thread.daemon = True
        thread.start()

    def register(self):
        name = self.entry_name.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        user_dir = os.path.join(DATASET_DIR, name)
        if os.path.exists(user_dir):
            messagebox.showerror("Error", "User already exists.")
            return
        os.makedirs(user_dir)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam. Please check your camera connection.")
            return

        count = 0
        max_images = 30

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                count += 1
                face_img = gray[y:y+h, x:x+w]
                img_path = os.path.join(user_dir, f"{count}.jpg")
                cv2.imwrite(img_path, face_img)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Images Captured: {count}/{max_images}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Register - Press 'q' to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if count >= max_images:
                break

        cap.release()
        cv2.destroyAllWindows()

        if count >= max_images:
            self.train_model()
            log_action("Registered", name)
            messagebox.showinfo("Success", f"User '{name}' registered with {count} images.")
        else:
            messagebox.showerror("Error", "Registration incomplete. Not enough images captured.")
            # Remove incomplete user data
            for f in os.listdir(user_dir):
                os.remove(os.path.join(user_dir, f))
            os.rmdir(user_dir)

    def train_model(self):
        faces, labels = [], []
        label_map = {}
        current_label = 0

        for user_name in os.listdir(DATASET_DIR):
            user_path = os.path.join(DATASET_DIR, user_name)
            if not os.path.isdir(user_path):
                continue
            label_map[current_label] = user_name
            for image_name in os.listdir(user_path):
                image_path = os.path.join(user_path, image_name)
                img = Image.open(image_path).convert('L')  # grayscale
                img_np = np.array(img, 'uint8')
                faces.append(img_np)
                labels.append(current_label)
            current_label += 1

        if len(faces) == 0:
            return

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save(MODEL_FILE)
        self.label_map = label_map

    def login_user(self):
        if not hasattr(self, 'label_map'):
            # Load label map from dataset folder
            self.label_map = {}
            current_label = 0
            for user_name in os.listdir(DATASET_DIR):
                user_path = os.path.join(DATASET_DIR, user_name)
                if os.path.isdir(user_path):
                    self.label_map[current_label] = user_name
                    current_label += 1
            if not self.label_map:
                messagebox.showerror("Error", "No registered users found. Please register first.")
                return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam. Please check your camera connection.")
            return

        recognized_name = "Unknown"
        confidence_threshold = 70  # Lower is more confident
        recognized = False
        max_frames = 100
        frame_count = 0

        while frame_count < max_frames and not recognized:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                label, confidence = self.recognizer.predict(face_img)
                if confidence < confidence_threshold:
                    recognized_name = self.label_map.get(label, "Unknown")
                    recognized = True
                    log_action("Login", recognized_name)
                    break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()

        if recognized:
            messagebox.showinfo("Success", f"Successfully logged in! Welcome, {recognized_name}")
        else:
            messagebox.showerror("Failed", "Face not recognized.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialAuthApp(root)
    root.mainloop()
