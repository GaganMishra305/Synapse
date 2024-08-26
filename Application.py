import streamlit as st
import cv2

from Utils.FaceUtils import detect_face_mesh, detect_haar_face, predict_emotion_cnn, predict_emotion_rf
from Utils.HandUtils import hand_detection_classification_rf, hand_detection_classification_cnn


#APP: Basic app settings
st.set_page_config(layout="wide")
st.title("Human Machine Interaction Organizer")
st.write("Select a model from the sidebar to switch models.")

#APP: Models selection sidebar
model_options = [
    "CNN Emotion Classifier (FRE)",
    "RF Emotion Classifier (Custom)",
    "CNN Hand Gesture Classifier (ASL)",
    "RF Gesture Classifier (Custom)",
    "Image Stylizer",
    "Caption generator",
]
st.sidebar.title("Model Selection")
selected_models = {name: st.sidebar.checkbox(name) for name in model_options}



#APP: run checkbox
run = st.checkbox('Run')
with st.container():
    FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

#APP: run opencv main screen
while run:
    ret, frame = camera.read()
    frame_height, frame_width, _ = frame.shape
    if not ret:
        st.warning("Failed to capture image. Please check your webcam.")
        break
    
    
    #APP: implementing a huge number of models(not so huge maybe lol...)
    if selected_models['CNN Emotion Classifier (FRE)']:
        faces = detect_haar_face(frame)
        if len(faces) > 0: 
            predict_emotion_cnn(frame, faces)
        
    if selected_models['RF Emotion Classifier (Custom)']:
        landmarks = detect_face_mesh(frame)
        if len(landmarks) ==1404:
            predict_emotion_rf(frame, landmarks)
    
    if selected_models['CNN Hand Gesture Classifier (ASL)']:
        hand_detection_classification_cnn(frame)
        
    if selected_models['RF Gesture Classifier (Custom)']:
        hand_detection_classification_rf(frame)
        

    
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Model output", width=1000)
    
camera.release()