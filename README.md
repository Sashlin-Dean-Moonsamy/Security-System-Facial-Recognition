# Facial Recognition Security System

A secure facial recognition system that can identify individuals and verify their visa/residency status in real-time using webcam input.

## Features

- Real-time face detection and recognition
- Support for both pretrained and untrained face recognition models
- Secure storage of face embeddings using encryption
- Visa/residency status verification
- Real-time display of recognition results with bounding boxes
- Support for multiple identities

## Prerequisites

- Python 3.7 or higher
- Webcam
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone Sashlin-Dean-Moonsamy/Security-System-Facial-Recognition
cd Security-System-Facial-Recog
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Adding New Faces

To add a new face to the system:
1. Place the face image in the project directory
2. Run the add_face.py script:
```bash
python add_face.py
```

### Running the System

To start the facial recognition system:
```bash
python main.py
```

- Press 'q' to quit the application
- The system will display real-time recognition results with:
  - Face bounding boxes
  - Identity information
  - Visa/residency status

## Project Structure

- `main.py` - Main application entry point
- `pretrained_fr.py` - Pretrained face recognition implementation
- `untrained_fr.py` - Custom face recognition implementation
- `add_face.py` - Script for adding new faces to the system
- `requirements.txt` - Project dependencies
- `embeddings.npz` - Encrypted storage for face embeddings
- `fernet.key` - Encryption key for secure storage

## Dependencies

- opencv-python - Computer vision and image processing
- numpy - Numerical computing
- deepface - Deep learning-based face recognition
- scikit-learn - Machine learning utilities
- cryptography - Secure storage implementation

## Security Features

- Face embeddings are encrypted using Fernet symmetric encryption
- Secure storage of identity information
- Real-time verification of visa/residency status

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Sashlin Dean Moonsamy
