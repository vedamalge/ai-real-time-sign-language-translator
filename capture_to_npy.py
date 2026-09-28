import cv2
import mediapipe as mp
import numpy as np
import os

# -----------------------
# Config
# -----------------------
GESTURES = [
    # Basic Greetings
    "Hello", "Goodbye", "ThankYou", "Please",
    
    # Common Responses
    "Yes", "No", "Maybe", "OK",
    
    # Basic Needs
    "Help", "Water", "Food", "Bathroom",
    
    # Emotions
    "Happy", "Sad", "Love", "Sorry",
    
    # Time Related
    "Today", "Tomorrow", "Later", "Now",
    
    # Questions
    "What", "Where", "When", "How",
    
    # Numbers
    "One", "Two", "Three", "Four", "Five",
    
    # Common Actions
    "Want", "Need", "Give", "Take",
    
    # Family
    "Mother", "Father", "Family", "Friend"
]

SAMPLES_PER_GESTURE = 50
DATA_PATH = "dataset"
os.makedirs(DATA_PATH, exist_ok=True)

# -----------------------
# MediaPipe Hands
# -----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

all_landmarks = []
all_labels = []

# -----------------------
# Capture function
# -----------------------
cap = cv2.VideoCapture(0)

for label_idx, gesture in enumerate(GESTURES):
    print(f"Prepare to record gesture: {gesture}")
    input("Press Enter to start...")

    count = 0
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            all_landmarks.append(landmarks)
            all_labels.append(label_idx)
            count += 1
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"{gesture} {count}/{SAMPLES_PER_GESTURE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# -----------------------
# Save dataset
# -----------------------
np.save(os.path.join(DATA_PATH, "landmarks.npy"), np.array(all_landmarks))
np.save(os.path.join(DATA_PATH, "labels.npy"), np.array(all_labels))
print("Landmarks and labels saved in dataset folder")
