import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet


class SecureFaceRecognition:
    """
    Secure face recognition using DeepFace (FaceNet) for embeddings
    and Fernet for encrypted embedding storage.
    """

    def __init__(self, threshold=0.6, cache_file="embeddings.npz", key_file="fernet.key"):
        self.threshold = threshold
        self.cache_file = cache_file
        self.key_file = key_file

        self.key = self.load_or_create_key()
        self.cipher_suite = Fernet(self.key)
        self.model = DeepFace.build_model("Facenet")
        self.embeddings_cache = []

        self.load_cache()

    def detect_faces(self, frame):
        """
        Detect faces in a frame using MTCNN via DeepFace.
        Returns a list of (face image, bounding box) tuples.
        """
        try:
            detected_faces = DeepFace.extract_faces(
                frame, detector_backend='mtcnn', enforce_detection=False
            )

            faces_info = []
            for face in detected_faces:
                region = face['facial_area']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                face_crop = frame[y:y+h, x:x+w]
                faces_info.append((face_crop, (x, y, w, h)))

            return faces_info
        except Exception as e:
            print(f"Error during face detection: {e}")
            return []

    def preprocess_frame(self, face_img):
        """Resize and normalize face image for embedding."""
        resized = cv2.resize(face_img, (160, 160))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def embed_face(self, face_img):
        """Generate face embedding using FaceNet."""
        result = DeepFace.represent(
            face_img, model_name="Facenet", enforce_detection=False
        )
        return result[0]['embedding'] if result else None

    def find_match(self, new_embedding):
        """Find best match in the cache using cosine similarity."""
        best_score = -1
        match_identity = None
        match_expiry = None

        for identity, embeddings in self.embeddings_cache:
            for encrypted_embedding, expiry in embeddings:
                decrypted = self.decrypt_embedding(encrypted_embedding)
                score = cosine_similarity([new_embedding], [decrypted])[0][0]

                if score > best_score and score >= self.threshold:
                    best_score = score
                    match_identity = identity
                    match_expiry = expiry

        return (match_identity, match_expiry), best_score

    def add_embedding(self, embedding, identity_info):
        """Add new encrypted embedding to cache."""
        identity, expiry = identity_info
        encrypted = self.encrypt_embedding(embedding)

        for entry in self.embeddings_cache:
            if entry[0] == identity:
                entry[1].append((encrypted, expiry))
                break
        else:
            self.embeddings_cache.append([identity, [(encrypted, expiry)]])

        self.save_cache()
        print(f"✅ Added identity: {identity} (Total identities: {len(self.embeddings_cache)})")

    def encrypt_embedding(self, embedding):
        """Encrypt embedding using Fernet."""
        array = np.array(embedding, dtype=np.float32)
        return self.cipher_suite.encrypt(array.tobytes())

    def decrypt_embedding(self, encrypted):
        """Decrypt Fernet-encrypted embedding."""
        decrypted = self.cipher_suite.decrypt(encrypted)
        return np.frombuffer(decrypted, dtype=np.float32)

    def save_cache(self):
        """Persist embedding cache to disk."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        print(f"💾 Saved {len(self.embeddings_cache)} identity(ies) to cache.")

    def load_cache(self):
        """Load embeddings from disk if available."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"📂 Loaded {len(self.embeddings_cache)} identity(ies) from cache.")
        else:
            print("🚫 No cache file found.")

    def clear_cache(self):
        """Clear in-memory and on-disk cache."""
        self.embeddings_cache = []
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("🚫 Cache cleared.")

    def list_identities(self):
        """List cached identities."""
        names = [identity for identity, _ in self.embeddings_cache]
        if names:
            print(f"🧠 Cached identities: {', '.join(set(names))}")
        else:
            print("🧠 No identities cached.")

    def load_or_create_key(self):
        """Load or create a Fernet encryption key."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        key = Fernet.generate_key()
        with open(self.key_file, 'wb') as f:
            f.write(key)
        return key
