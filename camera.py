import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create output file
os.makedirs("data", exist_ok=True)
csv_path = "data/hand_gestures.csv"

# Create CSV header if not exists
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("Press 1–5 to record a gesture, or ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Capture", frame)
    key = cv2.waitKey(1) & 0xFF

    # If a digit key is pressed, record the landmarks
    if key in [ord(str(i)) for i in range(1, 6)] and result.multi_hand_landmarks:
        label = int(chr(key))
        hand_landmarks = result.multi_hand_landmarks[0]
        row = [label]
        for lm in hand_landmarks.landmark:
            row += [lm.x, lm.y, lm.z]

        with open(csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow(row)

        print(f"Recorded gesture {label}")

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
