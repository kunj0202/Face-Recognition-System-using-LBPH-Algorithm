import cv2
import numpy as np
import os
import serial
import time

# Initialize serial communication with Arduino
arduino = serial.Serial('COM7', 9600)  
time.sleep(2)  # Wait for the connection to establish

# Function to detect and extract the face from the frame
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

# Dataset Collection
def collect_faces(data_path):
    cap = cv2.VideoCapture(0)
    person_name = input("Enter the name of the person: ")  # Prompt for the person's name
    count = 0

    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (300, 300))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save the image in the dataset folder with the person's name
            file_name_path = os.path.join(data_path, f'{person_name}_{count}.jpg')
            cv2.imwrite(file_name_path, face)

            # Display the image with the count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)
        else:
            print("Face not found")

        if cv2.waitKey(1) == 13 or count == 100:  # Press Enter key or capture 100 images
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection for {person_name} completed")


# Model Training
def train_model(data_path):
    onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    training_data, labels = [], []
    label_map = {}
    label_count = 0

    for file in onlyfiles:
        person_name = file.split('_')[0]  # Get person's name from filename
        image_path = os.path.join(data_path, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Assign unique label to each person based on their name
        if person_name not in label_map:
            label_map[person_name] = label_count
            label_count += 1

        label = label_map[person_name]
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(label)

    labels = np.asarray(labels, dtype=int)

    # Train the face recognition model using the LBPH algorithm
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data), np.asarray(labels))

    print("Model training completed")
    print("Label map:", label_map)

    return model, label_map


# Face Recognition and Prediction
def unlock_door(person_name):
    # Simulate door unlocking (this is where hardware control would go)
    print(f"Door unlocked for {person_name}!")  # Replace with actual unlocking logic

def face_detector(img, face_classifier, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
        return img, roi


def recognize_faces(model, label_map):
    cap = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    is_paused = False

    while True:
        if not is_paused:
            ret, frame = cap.read()
            image, face = face_detector(frame, face_classifier)

            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)

                if result[1] < 500:
                    confidence = int(100 * (1 - (result[1] / 300)))
                    person_name = [name for name, label in label_map.items() if label == result[0]][0]

                    if confidence > 80:
                        cv2.putText(image, person_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        unlock_door(person_name)

                        # Turn on green LED for recognized face
                        arduino.write(b'1')  # Send '1' to Arduino
                        is_paused = True  # Pause the video feed after successful recognition
                        print("Video feed paused")
                    else:
                        cv2.putText(image, "UNKNOWN", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                        # Turn on red LED for unrecognized face
                        arduino.write(b'0')  # Send '0' to Arduino

                cv2.imshow('Face Cropper', image)

            except:
                cv2.putText(frame, "FACE NOT FOUND", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Cropper', frame)

        else:
            cv2.putText(frame, "SYSTEM PAUSED. PRESS 'R' TO RESUME", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', frame)

        key = cv2.waitKey(1)
        if key == 32:  # Space to exit
            break
        elif key == ord('r') and is_paused:  # 'R' key to resume
            is_paused = False
            print("Video feed resumed")

    cap.release()
    cv2.destroyAllWindows()


# Main Program Flow
if __name__ == "__main__":
    dataset_path = 'C:/Users/kunjs/OneDrive/Desktop/dataset/'
    
    # Step 1: Collect dataset for yourself (e.g., Kunj)
    '''collect_faces(dataset_path)

    # Step 2: Collect dataset for your friend (e.g., John)
    collect_faces(dataset_path)'''

    # Step 3: Train the model with both your and your friend's data
    model, label_map = train_model(dataset_path)

    # Step 4: Start recognizing faces (will recognize both you and your friend)
    recognize_faces(model, label_map)

    # Close the serial connection
    arduino.close()
