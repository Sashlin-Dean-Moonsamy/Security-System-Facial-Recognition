# -----------------------------
# pretrained_fr.py
# -----------------------------
import cv2
import numpy as np
import os
import pickle
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from cryptography.fernet import Fernet


class SecureFaceRecognition:
    """
    A secure face recognition system using DeepFace (FaceNet model) for face embedding and Fernet for encrypted embedding storage.

    Attributes:
        threshold (float): Cosine similarity threshold for face matching.
        cache_file (str): File path to store embeddings cache.
        key_file (str): File path to store encryption key.
        cipher_suite (Fernet): Encryption/decryption utility.
        model (object): DeepFace FaceNet model instance.
        embeddings_cache (list): In-memory cache of [identity, [(encrypted_embedding, expiry)]] pairs.
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

    def detect_face(self, frame):
        """Detect the largest face in the given frame using Haar cascades."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None, (0, 0, 0, 0)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        return frame[y:y + h, x:x + w], (x, y, w, h)

    def preprocess_frame(self, face_img):
        """Resize and normalize a face image for embedding."""
        resized = cv2.resize(face_img, (160, 160))
        normalized = resized / 255.0
        return np.expand_dims(normalized, axis=0)

    def embed_face(self, face_img):
        """Generate a face embedding using the FaceNet model."""
        return DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]['embedding']

    def find_match(self, new_embedding):
        """Find the most similar face embedding in the cache using cosine similarity."""
        min_dist = float('inf')
        match_identity = None
        match_expiry = None
        for identity, embeddings in self.embeddings_cache:
            for encrypted_embedding, expiry in embeddings:
                decrypted_embedding = self.decrypt_embedding(encrypted_embedding)
                dist = cosine_similarity([new_embedding], [decrypted_embedding])[0][0]
                if dist < min_dist and dist < self.threshold:
                    min_dist = dist
                    match_identity = identity
                    match_expiry = expiry
        return (match_identity, match_expiry), min_dist

    def add_embedding(self, embedding, identity_info):
        """Encrypt and add a face embedding to the cache with identity and visa/residency info."""
        identity, expiry = identity_info
        encrypted_embedding = self.encrypt_embedding(embedding)
        existing_identity = next((entry for entry in self.embeddings_cache if entry[1][0] == identity), None)
        if existing_identity:
            existing_identity[1].append((encrypted_embedding, expiry))
        else:
            self.embeddings_cache.append([identity, [(encrypted_embedding, expiry)]])
        self.save_cache()
        print(f"✅ Added identity: {identity} with {len(self.embeddings_cache)} faces.")

    def encrypt_embedding(self, embedding):
        """Encrypt a face embedding using Fernet."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        return self.cipher_suite.encrypt(embedding.tobytes())

    def decrypt_embedding(self, encrypted_embedding):
        """Decrypt an embedding previously encrypted with Fernet."""
        decrypted = self.cipher_suite.decrypt(encrypted_embedding)
        return np.frombuffer(decrypted, dtype=np.float32)

    def save_cache(self):
        """Persist the current embeddings cache to disk using pickle."""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.embeddings_cache, f)
        print(f"💾 Saved {len(self.embeddings_cache)} face(s) to cache.")

    def load_cache(self):
        """Load embeddings cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.embeddings_cache = pickle.load(f)
            print(f"📂 Loaded {len(self.embeddings_cache)} face(s) from cache.")
        else:
            print("🚫 No cache file found.")

    def list_identities(self):
        """Print a list of unique cached identities."""
        unique_names = list(set([identity for identity, _ in self.embeddings_cache]))
        print(f"🧠 Cached identities: {', '.join(unique_names) if unique_names else 'None'}")

    def clear_cache(self):
        """Clear the cache in memory and on disk."""
        self.embeddings_cache = []
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("🚫 Cache cleared.")

    def load_or_create_key(self):
        """Load an existing encryption key or create a new one if not found."""
        if os.path.exists(self.key_file):
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            return key
