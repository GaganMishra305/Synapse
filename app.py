import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras import regularizers

import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Function to load the model with unique layer names
@st.cache_resource
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

    model.load_weights(f'Models/FERmodels/model_{version}.h5')
    return model



# _____________________________________________________________________________________________________________________________________
# Available model types
model_options = [
    "Emotion Classifier",
    "Hand Gestrue Classifier",
    "Pose Classifier",
    "Image Stylizer",
    "Caption generator",
    "Storyteller"
]

# Sidebar for model selection
st.sidebar.title("Model Selection")
selected_models = {name: st.sidebar.checkbox(name) for name in model_options}
selected_model = next((model_options[name] for name, selected in selected_models.items() if selected), None)




emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Main content layout
st.title("Human-Machine Interaction Organizer")
st.write("Select a model from the sidebar to switch models.")

run = st.checkbox('Run')
with st.container():
    FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.warning("Failed to capture image. Please check your webcam.")
        break

    facecasc = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predicting face emotion
        # prediction = model.predict(cropped_img)
        # maxindex = int(np.argmax(prediction))
        # cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Model output", width=1000)

camera.release()

