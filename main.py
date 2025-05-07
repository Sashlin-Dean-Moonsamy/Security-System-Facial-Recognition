import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from pretrained_fr import SecureFaceRecognition
from datetime import datetime


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Secure Face Recognition System")
        self.root.geometry("400x300")

        self.recognizer = SecureFaceRecognition()

        # Create and place widgets in the window
        self.create_widgets()

    def create_widgets(self):
        # Add Image Button
        self.add_button = tk.Button(self.root, text="Add Identity", command=self.add_identity)
        self.add_button.pack(pady=20)

        # Recognize Image Button
        self.recognize_button = tk.Button(self.root, text="Recognize Face", command=self.recognize_face)
        self.recognize_button.pack(pady=20)

        # List Identities Button
        self.list_button = tk.Button(self.root, text="List Identities", command=self.list_identities)
        self.list_button.pack(pady=20)

        # Clear Cache Button
        self.clear_button = tk.Button(self.root, text="Clear Cache", command=self.clear_cache)
        self.clear_button.pack(pady=20)

        # Status label
        self.status_label = tk.Label(self.root, text="", anchor="w", justify=tk.LEFT)
        self.status_label.pack(pady=10, padx=10, fill=tk.X)

    def add_identity(self):
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
        identities = []
        for identity_info, _ in self.recognizer.embeddings_cache:
            identities.append(identity_info["name"])
        
        if identities:
            identities_str = "\n".join(set(identities))
            messagebox.showinfo("Stored Identities", f"Stored Identities:\n{identities_str}")
        else:
            messagebox.showinfo("Stored Identities", "No identities stored.")

    def clear_cache(self):
        self.recognizer.clear_cache()
        messagebox.showinfo("Cache Cleared", "All identities have been cleared.")


if __name__ == "__main__":
    import tkinter.simpledialog as simpledialog

    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
