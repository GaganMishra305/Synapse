import pickle
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# models used
face_detection_haarcascade = cv2.CascadeClassifier('Models/haarcascade_frontalface_default.xml')
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,min_detection_confidence=0.5)

emotion_detector_rf = ...
emotion_detector_cnn = ...

def detect_haar_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection_haarcascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        

def detect_face_mesh(frame):
    # detecting face landmarks
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imageRGB)
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