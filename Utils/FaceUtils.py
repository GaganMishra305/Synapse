import pickle
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras import regularizers

# models used
face_detection_haarcascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5)

# load self created models
with open('Models/FER models/RFmodels/model_v1.p', 'rb') as f:
    emotion_detector_rf = pickle.load(f)
def load_model(version):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48, 48, 1), name='conv2d_1'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2d_2'))
    model.add(BatchNormalization(name='batchnorm_1'))
    model.add(MaxPool2D(pool_size=(2, 2), name='maxpool_1'))
    model.add(Dropout(0.25, name='dropout_1'))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu', name='conv2d_3'))
    model.add(BatchNormalization(name='batchnorm_2'))
    model.add(MaxPool2D(pool_size=(2, 2), name='maxpool_2'))
    model.add(Dropout(0.25, name='dropout_2'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='conv2d_4'))
    model.add(BatchNormalization(name='batchnorm_3'))
    model.add(MaxPool2D(pool_size=(2, 2), name='maxpool_3'))
    model.add(Dropout(0.25, name='dropout_3'))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='conv2d_5'))
    model.add(BatchNormalization(name='batchnorm_4'))
    model.add(MaxPool2D(pool_size=(2, 2), name='maxpool_4'))
    model.add(Dropout(0.25, name='dropout_4'))

    model.add(Flatten(name='flatten_1'))
    model.add(Dense(256, activation='relu', name='dense_1'))
    model.add(BatchNormalization(name='batchnorm_5'))
    model.add(Dropout(0.25, name='dropout_5'))

    model.add(Dense(512, activation='relu', name='dense_2'))
    model.add(BatchNormalization(name='batchnorm_6'))
    model.add(Dropout(0.25, name='dropout_6'))

    model.add(Dense(7, activation='softmax', name='output'))

    model.load_weights(f'Models/FER models/CNNmodels/model_{version}.h5')
    return model
emotion_detector_cnn = load_model('v2')


def detect_haar_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection_haarcascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
    
    return faces
        

def detect_face_mesh(frame):
    # detecting face landmarks
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imageRGB)
    
    image_landmarks = []
    if results.multi_face_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
        
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=results.multi_face_landmarks[0],
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)
        
        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = []
        ys_ = []
        zs_ = []
        for idx in ls_single_face:
            xs_.append(idx.x)
            ys_.append(idx.y)
            zs_.append(idx.z)
        for j in range(len(xs_)):
            image_landmarks.append(xs_[j] - min(xs_))
            image_landmarks.append(ys_[j] - min(ys_))
            image_landmarks.append(zs_[j] - min(zs_))
            
    return image_landmarks

def predict_emotion_rf(frame, face_landmarks):
    emotions = ['Happy', 'Sad', 'Surprised']
    output = emotion_detector_rf.predict([face_landmarks])
    cv2.putText(frame, emotions[int(output[0])],(10, frame.shape[0] - 1),cv2.FONT_HERSHEY_SIMPLEX,3,(0, 255, 0),5)
    
def predict_emotion_cnn(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emotion_detector_cnn.predict(cropped_img, verbose=2)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)