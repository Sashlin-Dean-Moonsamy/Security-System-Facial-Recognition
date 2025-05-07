import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from pretrained_fr import SecureFaceRecognition
from datetime import datetime
from PIL import Image, ImageTk


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Face Recognition System")
        self.root.geometry("800x600")

        self.recognizer = SecureFaceRecognition()

        self.is_recognizing = False  # Flag to control the video feed
        self.video_source = 0  # Default webcam source

        # Create and place widgets
        self.create_widgets()

    def create_widgets(self):
        # Video Frame (to display the webcam feed)
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(pady=10)

        # Start Video Button
        self.start_video_button = tk.Button(self.root, text="Start Camera", command=self.start_video)
        self.start_video_button.pack(pady=10)

        # Add Image Button
        self.add_button = tk.Button(self.root, text="Add Identity", command=self.add_identity)
        self.add_button.pack(pady=10)

        # Recognize Image Button
        self.recognize_button = tk.Button(self.root, text="Recognize Face", command=self.recognize_face)
        self.recognize_button.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="", anchor="w", justify=tk.LEFT)
        self.status_label.pack(pady=10, padx=10, fill=tk.X)

    def start_video(self):
        """Start the webcam feed for real-time face recognition."""
        self.is_recognizing = True
        self.capture = cv2.VideoCapture(self.video_source)

        self.update_video_feed()

    def stop_video(self):
        """Stop the webcam feed."""
        self.is_recognizing = False
        if hasattr(self, 'capture'):
            self.capture.release()

    def update_video_feed(self):
        """Update the video feed in real-time."""
        if not self.is_recognizing:
            return

        ret, frame = self.capture.read()
        if ret:
            # Detect faces in the frame
            faces = self.recognizer.detect_faces(frame)

            for face_img, (x, y, w, h) in faces:
                # Highlight the detected face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Get embedding and recognize the identity
                embedding = self.recognizer.embed_face(face_img)
                if embedding is not None:
                    identity, score = self.recognizer.find_match(embedding)
                    if identity:
                        # Display name and expiry status
                        name = identity['name']
                        status = identity['status']
                        expiry = identity.get('expiry', 'N/A')
                        text = f"{name} ({status})"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Convert the frame to an image for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = ImageTk.PhotoImage(img)

            # Update the video feed in Tkinter window
            self.video_frame.img_tk = img
            self.video_frame.config(image=img)

        self.root.after(10, self.update_video_feed)  # Call this method after 10 ms

    def add_identity(self):
        """Add a new identity."""
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
        if not file_path:
            return

        # Get user input for identity details
        name = simpledialog.askstring("Identity", "Enter Name:")
        if not name:
            messagebox.showerror("Error", "Name cannot be empty!")
            return

        identity_type = simpledialog.askstring("Identity Type", "Enter Type (resident/visa):")
        if identity_type not in ["resident", "visa"]:
            messagebox.showerror("Error", "Type must be 'resident' or 'visa'!")
            return

        expiry = None
        if identity_type == "visa":
            expiry = simpledialog.askstring("Visa Expiry", "Enter Expiry Date (YYYY-MM-DD):")
            try:
                datetime.strptime(expiry, "%Y-%m-%d")
            except ValueError:
                messagebox.showerror("Error", "Expiry must be in YYYY-MM-DD format!")
                return

        # Read and process image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Unable to load image!")
            return

        faces = self.recognizer.detect_faces(img)
        if not faces:
            messagebox.showerror("Error", "No face found in the image!")
            return

        face_img, _ = faces[0]
        embedding = self.recognizer.embed_face(face_img)
        if embedding is None:
            messagebox.showerror("Error", "Embedding failed!")
            return

        identity_info = {
            "name": name,
            "type": identity_type,
            "expiry": expiry
        }
        self.recognizer.add_embedding(embedding, identity_info)
        messagebox.showinfo("Success", f"Identity '{name}' added successfully!")

    def recognize_face(self):
        """Recognize face from a file image."""
        # Open file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png")])
        if not file_path:
            return

        # Read and process image
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Unable to load image!")
            return

        faces = self.recognizer.detect_faces(img)
        if not faces:
            messagebox.showerror("Error", "No face detected!")
            return

        face_img, _ = faces[0]
        embedding = self.recognizer.embed_face(face_img)
        if embedding is None:
            messagebox.showerror("Error", "Embedding failed!")
            return

        identity, score = self.recognizer.find_match(embedding)
        if identity:
            status_msg = f"Name: {identity['name']}\nType: {identity['type']}\nStatus: {identity['status']}\n"
            if 'expiry' in identity:
                status_msg += f"Expiry Date: {identity['expiry']}\n"
            status_msg += f"Similarity Score: {score:.4f}"
            self.status_label.config(text=status_msg)
        else:
            self.status_label.config(text="No match found.")

    def list_identities(self):
        """List all stored identities."""
        identities = []
        for identity_info, _ in self.recognizer.embeddings_cache:
            identities.append(identity_info["name"])
        
        if identities:
            identities_str = "\n".join(set(identities))
            messagebox.showinfo("Stored Identities", f"Stored Identities:\n{identities_str}")
        else:
            messagebox.showinfo("Stored Identities", "No identities stored.")

    def clear_cache(self):
        """Clear all stored identities."""
        self.recognizer.clear_cache()
        messagebox.showinfo("Cache Cleared", "All identities have been cleared.")


if __name__ == "__main__":
    import tkinter.simpledialog as simpledialog

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
