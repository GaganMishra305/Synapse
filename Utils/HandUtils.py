import pickle
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import model_from_json # type: ignore

# models used
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hand_image = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1)
hand_video = mp_hands.Hands(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7,min_tracking_confidence=0.7, max_num_hands=1)

# loading models
model_dict = pickle.load(open('Models/HandGesture models/RFmodel/model_v3.p', 'rb'))
gesture_detector_rf = model_dict['model']

with open('Models/HandGesture models/CNNmodel/model_cnn_v2.json', 'r') as json_file:
    loaded_model_json = json_file.read()
gesture_detector_cnn = model_from_json(loaded_model_json)
gesture_detector_cnn.load_weights('Models/HandGesture models/CNNmodel/model_cnn_v2.h5')
gesture_detector_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def hand_detection_classification_rf(frame, draw=True):
    # detecting hand landmarks
    frame_height, frame_width, _ = frame.shape
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_video.process(imageRGB)
    if results.multi_hand_landmarks and draw:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                    )
            
        x_=[]
        y_=[]
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * frame_width) - 10
        y1 = int(min(y_) * frame_height) - 10

        x2 = int(max(x_) * frame_width) - 10
        y2 = int(max(y_) * frame_height) - 10
        
        if len(data_aux) == 42: 
            prediction = gesture_detector_rf.predict([np.asarray(data_aux)])
            predicted_character = prediction[0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
            
            


def hand_detection_classification_cnn(frame, draw=True):
    # detecting hand landmarks
    frame_height, frame_width, _ = frame.shape
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_video.process(imageRGB)
    if results.multi_hand_landmarks and draw:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                    )
            
        x_=[]
        y_=[]
        data_aux = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * frame_width) - 10
        y1 = int(min(y_) * frame_height) - 10

        x2 = int(max(x_) * frame_width) - 10
        y2 = int(max(y_) * frame_height) - 10
        
        if len(data_aux) == 42: 
            X = np.array(data_aux)
            X = X.reshape((1,42))
            prediction = gesture_detector_cnn.predict(X, verbose=0)
            predicted_character = np.argmax(prediction, axis=1)[0]
            
            asl_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

            predicted_character = asl_labels[predicted_character]
            # print('Prediciton: ',predicted_character)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)