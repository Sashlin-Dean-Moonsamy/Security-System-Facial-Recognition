# -----------------------------
# add_face.py
# -----------------------------

import cv2
import datetime
from pretrained_fr import SecureFaceRecognition  # Replace 'test' with the actual filename if different

def add_new_face(image_path, identity_info, face_recognition_system):
    identity, valid_until = identity_info

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Unable to load image {image_path}.")
        return

    # Detect the face in the image
    face, (x, y, w, h) = face_recognition_system.detect_face(image)
    if face is None:
        print(f"⚠️ Error: No face detected in the image {image_path}.")
        return

    # Create an embedding from the raw face image
    embedding = face_recognition_system.embed_face(face)

    # Add the embedding to the system
    face_recognition_system.add_embedding(embedding, (identity, valid_until))
    print(f"✅ New face added: {identity}, visa: {valid_until} from {image_path}")
    face_recognition_system.save_cache()


# Initialize the face recognition system
face_recognition_system = SecureFaceRecognition()
face_recognition_system.load_cache()

# User-provided values
identity = "sash"
is_resident = False  # Change to True if the person is a resident

if is_resident:
    valid_until = "resident"
else:
    valid_minutes = 1  # Change visa duration as needed
    valid_until = (datetime.datetime.now() + datetime.timedelta(minutes=valid_minutes)).isoformat()

# List of image paths for this user
image_paths = [".\me1.jpg", ".\me2.jpg", ".\me3.jpg"]  # Add all paths to images you want for the user

# Add each face image to the system
for image_path in image_paths:
    add_new_face(image_path, (identity, valid_until), face_recognition_system)
