import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


DATA_DIR = "asl_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
num_samples = 100  

def extract_landmarks(hand_landmarks):
    """Extract flattened landmark coordinates (x, y, z)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)


cap = cv2.VideoCapture(0)
data = []
label_count = {label: 0 for label in labels}

print("Instructions: Show one letter at a time, referring to the ASL alphabet PNG. Press 'n' to capture a sample, 'q' to quit.")

current_label = labels[0]
print(f"Start with letter: {current_label}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
    cv2.putText(frame, f"Letter: {current_label} ({label_count[current_label]}/{num_samples})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Data Collection', frame)
    key = cv2.waitKey(1) & 0xFF

    
    if key == ord('n') and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            features = extract_landmarks(hand_landmarks)
            data.append(np.append(features, current_label))  
            label_count[current_label] += 1
            print(f"Captured {label_count[current_label]} samples for {current_label}")

        
        if label_count[current_label] >= num_samples:
            current_label = labels[(labels.index(current_label) + 1) % len(labels)] if labels.index(current_label) < len(labels) - 1 else None
            if current_label is None:
                print("Data collection complete!")
                break
            print(f"Now collecting for letter: {current_label}")

    
    if key == ord('q'):
        break


columns = [f"lm_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data, columns=columns)
df.to_csv(os.path.join(DATA_DIR, 'asl_dataset_a_z.csv'), index=False)


cap.release()
cv2.destroyAllWindows()
hands.close()

print(f"Dataset saved to {os.path.join(DATA_DIR, 'asl_dataset_a_z.csv')}")