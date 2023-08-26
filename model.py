import cv2
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from expressive_eyes.face_manager import FaceManager
from expressive_eyes import expressions as exp

# Load the face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Load the emotion detection model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("fer.h5")
print("Loaded models from disk")

# Define the emotion labels
labels = ['Angry', 'Sad', 'Sad', 'Happy', 'Sad', 'Happy', 'Neutral']

# Define the names corresponding to face IDs
names = ['None', 'Mareez']

# Initialize and start the video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

fm = FaceManager()

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Face recognition
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        if id >= 0 and id < len(names):
            id = names[id]
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 255), 2)

        # Emotion detection
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1)
        cropped_img = cropped_img / 255.0

        # Predict the emotion label for the ROI
        yhat = emotion_model.predict(np.expand_dims(cropped_img, 0))
        emotion_label = labels[np.argmax(yhat)]
	if emotion_label = 'Happy': 
        cv2.putText(img, emotion_label, (x, y + h + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2), fm.set_next_expression(exp.get("Happy")

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break

# Cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

