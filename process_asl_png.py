import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from PIL import Image
import os
import traceback


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)


DATA_DIR = "asl_dataset"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def extract_landmarks(hand_landmarks):
    """Extract flattened landmark coordinates (x, y, z)."""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    return np.array(landmarks)

try:
    
    PNG_PATH = "asl_alphabet.png"  
    if not os.path.exists(PNG_PATH):
        raise FileNotFoundError(f"PNG file not found at {PNG_PATH}")

    img = Image.open(PNG_PATH)
    img_np = np.array(img)
    print(f"Loaded PNG: {img_np.shape}")

    
    rows, cols = 5, 6
    if img_np.shape[0] < rows or img_np.shape[1] < cols:
        raise ValueError(f"Image dimensions {img_np.shape} too small for {rows}x{cols} grid")

    height, width = img_np.shape[0] // rows, img_np.shape[1] // cols
    print(f"Sub-image size: {width}x{height}")

    data = []
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx >= 26:  
                break
            
            sub_img = img_np[i * height:(i + 1) * height, j * width:(j + 1) * width]
            
            try:
                if sub_img.shape[2] == 4:  
                    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_RGBA2RGB)
                else:
                    sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
            except IndexError:
                print(f"Skipping sub-image {idx} (letter {labels[idx]}): Invalid shape {sub_img.shape}")
                continue

            
            results = hands.process(sub_img)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    features = extract_landmarks(hand_landmarks)
                    data.append(np.append(features, labels[idx]))
                    print(f"Processed letter {labels[idx]}")
            else:
                print(f"No hand detected in sub-image {idx} (letter {labels[idx]})")

    
    if data:
        columns = [f"lm_{i}_{coord}" for i in range(21) for coord in ['x', 'y', 'z']] + ['label']
        df = pd.DataFrame(data, columns=columns)
        output_path = os.path.join(DATA_DIR, 'asl_dataset_from_png.csv')
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path} with {len(df)} samples")
    else:
        print("No data collected. Check PNG structure or hand detection settings.")

except Exception as e:
    print(f"Error: {str(e)}")
    traceback.print_exc()

finally:

    hands.close()