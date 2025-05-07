# -----------------------------
# main.py
# -----------------------------
import cv2
from datetime import datetime
from pretrained_fr import SecureFaceRecognition

# Initialize the secure face recognition system
face_recognition = SecureFaceRecognition()
face_recognition.load_cache()
face_recognition.list_identities()

# Start webcam capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Recognition")

print("🔍 Starting face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
        .detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        processed = face_recognition.preprocess_frame(face)
        embedding = face_recognition.embed_face(face)

        identity_info, dist = face_recognition.find_match(embedding)
        print(f"📏 Similarity score: {dist:.4f}")

        if identity_info:
            identity, expiry = identity_info
            if expiry == "resident":
                visa_status = "Resident"
            else:
                if isinstance(expiry, str):
                    try:
                        expiry_dt = datetime.fromisoformat(expiry)
                        visa_status = "Valid Visa" if expiry_dt > datetime.now() else "Expired Visa"
                    except ValueError:
                        visa_status = "Invalid Expiry Date"
                        print(f"⚠️ Invalid expiry date format for {identity}. Setting as 'Invalid Expiry Date'.")
                else:
                    visa_status = "Unknown Expiry"
                    print(f"⚠️ Expiry information for {identity} is not valid.")

            text = f"{identity} - {visa_status}"

        # Draw face box and identity
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    # Display the frame
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(50) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        print("👋 Exiting...")
        break

# Cleanup
cap.release()
face_recognition.save_cache()
cv2.destroyAllWindows()
