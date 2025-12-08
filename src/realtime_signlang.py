import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from collections import deque

MODEL_PATH = os.path.join("models", "sign_model.pkl")
LABEL_ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError("Model or label encoder not found. Train the model first.")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder


def extract_landmarks(hand_landmarks):
    x_coords = []
    y_coords = []
    z_coords = []

    for lm in hand_landmarks.landmark:
        x_coords.append(lm.x)
        y_coords.append(lm.y)
        z_coords.append(lm.z)

    return np.array(x_coords + y_coords + z_coords).reshape(1, -1)

def main():
    model, label_encoder = load_model()

    cap = cv2.VideoCapture(0)
    predicted_letter = ""
    word = ""
    
    # store last few predictions for smoothing
    preds_queue = deque(maxlen=7)
    last_letter_added = False

    with mp_hands.Hands(max_num_hands=1) as hands:
        print("Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                features = extract_landmarks(lm)
                pred = model.predict(features)[0]
                predicted_letter = label_encoder.inverse_transform([pred])[0]

                preds_queue.append(predicted_letter)

                # Stable prediction using frequency
                if len(preds_queue) == preds_queue.maxlen:
                    stable = max(set(preds_queue), key=preds_queue.count)

                    cv2.putText(frame, f"Letter: {stable}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Add letter to word only once when hand disappears
                    last_letter_added = False

            else:
                # hand disappears -> confirm latest stable prediction
                if len(preds_queue) > 0 and not last_letter_added:
                    stable = max(set(preds_queue), key=preds_queue.count)
                    word += stable
                    preds_queue.clear()
                    last_letter_added = True

            # Display typed word
            cv2.putText(frame, f"Word: {word}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.imshow("Sign Language Word Builder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            if key == ord(' '):
                word += " "
            if key == 8:  # Backspace key
                word = word[:-1]
            if key == 13:  # Enter key
                print("Final sentence:", word)
                word = ""

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
