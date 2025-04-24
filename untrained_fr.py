# -----------------------------
# fr.py
# -----------------------------
import numpy as np
import cv2
import os

class FaceRecognition:
    def __init__(self, threshold=0.6):
        self.embeddings_cache = []  # Stores (embedding, identity, visa_expiry)
        self.threshold = threshold
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.num_filters = 8
        self.kernel_size = 3
        self.kernels_path = "kernels.npy"
        self.fc_weights_path = "fc_weights.npy"

        conv_output_size = 62
        pooled_output_size = conv_output_size // 2
        self.flat_dim = self.num_filters * pooled_output_size * pooled_output_size

        # Load or initialize model weights
        self.kernels = self.load_or_init(self.kernels_path, (self.num_filters, self.kernel_size, self.kernel_size))
        self.fc_weights = self.load_or_init(self.fc_weights_path, (128, self.flat_dim))

    def load_or_init(self, path, shape):
        if os.path.exists(path):
            print(f"🔁 Loaded model weights from {path}")
            return np.load(path)
        else:
            print(f"🧠 Initializing new weights: {path}")
            weights = np.random.randn(*shape) * 0.01
            np.save(path, weights)
            return weights

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None, (0, 0, 0, 0)
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        return frame[y:y+h, x:x+w], (x, y, w, h)

    def preprocess_frame(self, frame):
        gray_frame = self.to_grayscale(frame)
        resized = self.resize_frame(gray_frame, (64, 64))
        normalized = resized / 255.0
        return normalized

    def to_grayscale(self, frame):
        return np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])

    def resize_frame(self, frame, target_size):
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def embed_face(self, face_image):
        if face_image.shape != (64, 64):
            raise ValueError("Input image must be 64x64")

        conv_output = np.zeros((self.num_filters, 62, 62))
        for f in range(self.num_filters):
            kernel = self.kernels[f]
            for i in range(62):
                for j in range(62):
                    region = face_image[i:i+3, j:j+3]
                    conv_output[f, i, j] = np.sum(region * kernel)

        relu_output = np.maximum(conv_output, 0)

        pooled_output = np.zeros((self.num_filters, 31, 31))
        for f in range(self.num_filters):
            for i in range(31):
                for j in range(31):
                    region = relu_output[f, i*2:i*2+2, j*2:j*2+2]
                    pooled_output[f, i, j] = np.max(region)

        flat = pooled_output.reshape(-1)
        fc_output = np.dot(self.fc_weights, flat)
        norm = np.linalg.norm(fc_output)
        return fc_output if norm == 0 else fc_output / norm

    def find_match(self, embedding):
        min_dist = np.inf
        match_identity = None
        expiry = None
        for cached_embedding, identity, valid_until in self.embeddings_cache:
            dist = np.linalg.norm(cached_embedding - embedding)
            print(f"📏 Distance to {identity}: {dist:.4f}")
            if dist < min_dist:
                min_dist = dist
                match_identity = identity
                expiry = valid_until
        return match_identity, min_dist, expiry

    def add_embedding(self, embedding, identity_info):
        identity, valid_until = identity_info
        self.embeddings_cache.append((embedding, identity.strip(), valid_until))
        print(f"✅ Added identity: {identity} (visa: {valid_until})")

    def save_cache(self, filename='embeddings.npz'):
        embeddings = [e for e, _, _ in self.embeddings_cache]
        identities = [i for _, i, _ in self.embeddings_cache]
        expiries = [v for _, _, v in self.embeddings_cache]
        np.savez(filename, embeddings=np.array(embeddings, dtype=object),
                 identities=np.array(identities),
                 expiries=np.array(expiries, dtype=object))
        print(f"💾 Saved {len(embeddings)} face(s) to cache.")

    def load_cache(self, filename='embeddings.npz'):
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True)
            embeddings = data['embeddings']
            identities = data['identities']
            expiries = data['expiries']
            self.embeddings_cache = list(zip(embeddings, identities, expiries))
            print(f"📂 Loaded {len(self.embeddings_cache)} face(s) from cache.")
        else:
            print("🚫 No cache file found.")

    def list_identities(self):
        unique_names = list(set([id for _, id, _ in self.embeddings_cache]))
        print(f"🧠 Cached identities: {', '.join(unique_names) if unique_names else 'None'}")
