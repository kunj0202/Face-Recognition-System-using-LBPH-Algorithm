# Face Recognition System using LBPH Algorithm

A simple face recognition project developed using **Python** and **OpenCV**. This application uses the **Local Binary Patterns Histograms (LBPH)** algorithm to recognize faces in real time through a webcam. It allows users to create their own face dataset, train the recognition model, and identify registered individuals during live video capture.

---

## Overview

The project is divided into three main stages:

- **Dataset Collection** – Captures multiple face images of a user through the webcam.
- **Model Training** – Trains an LBPH face recognizer using the collected images.
- **Real-Time Recognition** – Detects and recognizes registered faces from a live webcam feed.

The project is intended as a beginner-friendly implementation of a complete face recognition pipeline using classical computer vision techniques.

---

## Features

- Capture face images directly from a webcam
- Face detection using Haar Cascade Classifier
- Face recognition using the LBPH algorithm
- Automatic label assignment for multiple users
- Real-time recognition with confidence-based prediction
- Simple pause and resume functionality after successful recognition

---

## Technologies Used

- Python
- OpenCV (opencv-contrib-python)
- NumPy

---

## Project Structure

```
Face-Recognition-System-using-LBPH/
│
├── dataset/                              # Stores captured face images
├── haarcascade_frontalface_default.xml   # Face detection model
├── main.py                               # Main source code
├── requirements.txt
└── README.md
```

---

## How It Works

### 1. Dataset Collection

The first step is collecting images of each person you want the system to recognize.

After entering the person's name, the webcam captures around **100 facial images**. Each detected face is cropped, converted to grayscale, resized, and stored inside the dataset folder.

Example:

```
John_1.jpg
John_2.jpg
John_3.jpg
Alice_1.jpg
Alice_2.jpg
```

---

### 2. Model Training

Once the dataset is ready, the program reads all stored images and automatically assigns labels based on the filenames.

The images are then used to train an **LBPH Face Recognizer**, which learns the facial features of each registered person.

---

### 3. Face Recognition

During recognition, the webcam continuously scans for faces.

Whenever a face is detected:

- The face is extracted and converted to grayscale.
- The trained LBPH model predicts the person's identity.
- If the confidence score is high enough, the person's name is displayed on the screen.
- The video pauses after a successful recognition and can be resumed by pressing **R**.

Press **Spacebar** at any time to exit the application.

---

## Installation

Clone this repository:

```bash
git clone https://github.com/your-username/Face-Recognition-System-using-LBPH.git
```

Move into the project folder:

```bash
cd Face-Recognition-System-using-LBPH
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install opencv-contrib-python numpy
```

---

## Running the Project

### Step 1 – Collect Face Images

Uncomment the following line inside `main.py`:

```python
collect_faces(dataset_path)
```

Run the program:

```bash
python main.py
```

Enter the person's name and allow the application to capture the images.

---

### Step 2 – Train the Model

The model is trained automatically using the images stored in the dataset folder.

---

### Step 3 – Start Face Recognition

Run the program again:

```bash
python main.py
```

The webcam will open and begin recognizing registered faces in real time.

---

## Controls

| Key | Action |
|------|--------|
| **R** | Resume recognition after a successful detection |
| **Spacebar** | Exit the application |

---

## Why LBPH?

The **Local Binary Patterns Histograms (LBPH)** algorithm is a popular face recognition technique because it is lightweight, easy to implement, and performs well even with relatively small datasets. Unlike deep learning models, LBPH does not require powerful hardware or extensive training data, making it an excellent choice for learning computer vision fundamentals and building small-scale face recognition applications.

---

## Future Improvements

Some possible enhancements for this project include:

- Saving the trained model instead of retraining it every time the program starts
- Improving recognition accuracy by tuning confidence thresholds
- Supporting recognition of multiple faces simultaneously
- Adding a graphical user interface (GUI)
- Logging recognized users with timestamps
- Integrating the system with an Arduino or ESP32-based smart door lock
- Replacing Haar Cascade with more robust face detectors such as MTCNN or DNN-based detectors

---

## Author

**Kunj Bihari Sharma**

---

If you found this project useful or learned something from it, feel free to leave a ⭐ on the repository. It would be greatly appreciated!
