import cv2
import numpy as np
import os

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
    
    person_name = input("Enter the name of the person: ")  
    print(f"Collecting data for {person_name}. Please look at the camera.")
    
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if ret:  
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (300, 300))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                
                file_name_path = os.path.join(data_path, f'{person_name}_{count}.jpg')
                cv2.imwrite(file_name_path, face)

                
                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not found")
        else:
            print("Failed to capture frame")

        
        if cv2.waitKey(1) == 13 or count == 100:  
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
        person_name = file.split('_')[0]  
        image_path = os.path.join(data_path, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        
        if person_name not in label_map:
            label_map[person_name] = label_count
            label_count += 1

        label = label_map[person_name]
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(label)

    labels = np.asarray(labels, dtype=int)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(training_data), np.asarray(labels))

    print("Model training completed")
    '''print("Label map:", label_map)'''

    return model, label_map


# Face Recognition and Prediction
'''def unlock_door(person_name):
    print(f"Door unlocked for {person_name}!")'''


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
            if not ret:
                print("Failed to capture frame")
                break

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
                        is_paused = True  
                        print("Video feed paused")
                    else:
                        cv2.putText(image, "UNKNOWN", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Face Cropper', image)

            except:
                '''cv2.putText(frame, "FACE NOT FOUND", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)'''
                cv2.imshow('Face Cropper', frame)

        else:
            cv2.putText(frame, "SYSTEM PAUSED. PRESS 'R' TO RESUME", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', frame)

        key = cv2.waitKey(1)
        if key == 32:  
            break
        elif key == ord('r') and is_paused:  
            is_paused = False
            print("Video feed resumed")

    cap.release()
    cv2.destroyAllWindows()


# Main Program Flow
if __name__ == "__main__":
    dataset_path = 'C:/Users/kunjs/OneDrive/Desktop/dataset/'
    
    
    '''collect_faces(dataset_path)'''

    
    '''collect_faces(dataset_path)'''

    
    model, label_map = train_model(dataset_path)

    
    recognize_faces(model, label_map)
