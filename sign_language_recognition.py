import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


try:
    with open('asl_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Model file not found. Please train and save a model first.")
    model = None


label_map = {i: chr(65 + i) for i in range(26)}  


cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Extract flattened landmark coordinates (x, y, z)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = hands.process(frame_rgb)

    predicted_label = "None"
    if results.multi_hand_landmarks and model is not None:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            features = extract_landmarks(hand_landmarks)

            
            try:
                prediction = model.predict([features])[0]
                predicted_label = label_map.get(prediction, "Unknown")
            except Exception as e:
                predicted_label = f"Error: {str(e)}"

    
    cv2.putText(frame, f"Sign: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    
    cv2.imshow('Sign Language Recognition', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hands.close()