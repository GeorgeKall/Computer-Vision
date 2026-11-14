import cv2
import mediapipe as mp
import pickle
import numpy as np
import torch
from model_class import MyModel 

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but KNeighborsClassifier was fitted with feature names"
)
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Load trained model
with open("data/model.pkl", "rb") as f:
    model = MyModel()
    model.load_state_dict(torch.load(f, weights_only=True))
    model.eval()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(2) #change here which camera we are using if system has multiple

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        prediction = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark positions into a flat list
                data_row = []
                for lm in hand_landmarks.landmark:
                    data_row += [lm.x, lm.y, lm.z]

                # Predict gesture
                data_row = np.array(data_row).reshape(1, -1)
                # Make input into a tensor
                data_tensor = torch.from_numpy(data_row).to(torch.float32)
                prediction = int(torch.argmax(torch.softmax(model(data_tensor), dim=1)))
                prediction = prediction + 1 # add 1 to the prediction cause we want to predict labels 1-5 


        # Show prediction on screen
        if prediction:
            cv2.putText(
                frame,
                f"Prediction: {prediction}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
                cv2.LINE_AA,
            )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break  # ESC to quit

cap.release()
cv2.destroyAllWindows()
