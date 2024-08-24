import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten,Dense,Dropout,BatchNormalization
from keras import regularizers
from keras.models import model_from_json



json_file = open("Models/FERmodels/model_v3.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("Models/FERmodels/model_v3_weights.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

# emotion_dict = {0:'angry', 1 :'happy', 2: 'neutral', 3:'sad', 4: 'surprise'}
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

# ______________________________________________________________________________________________________________________________________________

# Streamlit app
st.title("Real-time Emotion Detection")

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture image. Please check your webcam.")
        break
    
    # face_cascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    
    for (x, y, w, h) in faces:
        image = gray[x:x+w,y:y+h]
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        image = cv2.resize(image,(48,48))
        img = extract_features(image)
        
        # prediciting
        pred = model.predict(img)
        prediction_label = emotion_dict[pred.argmax()]
        # print(pred)
        cv2.putText(frame, '% s' %(prediction_label), (x-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
camera.release()
