import cv2
import mediapipe as mp
import csv
import os

# ====== CONFIG ======
CURRENT_LETTER = "Z"  # <-- CHANGE THIS when collecting for B, C, etc.
OUTPUT_CSV = os.path.join("data", "sign_data.csv")
# =====================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def create_csv_if_not_exists():
    # If file doesn't exist, create it with header
    if not os.path.exists(OUTPUT_CSV):
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        with open(OUTPUT_CSV, mode="w", newline="") as f:
            writer = csv.writer(f)
            # header: label + 63 coords (21 landmarks * 3)
            header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
            writer.writerow(header)

def main():
    create_csv_if_not_exists()

    cap = cv2.VideoCapture(0)  # 0 = default webcam

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        print(f"Collecting data for letter: {CURRENT_LETTER}")
        print("Press 'C' to capture a sample, 'Q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # mirror view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks on the video
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Instructions overlay
            cv2.putText(frame, f"Letter: {CURRENT_LETTER}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'C' to capture, 'Q' to quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Data Collection - Sign Language", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture current frame landmarks
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]

                    x_coords = []
                    y_coords = []
                    z_coords = []

                    for lm in hand_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)
                        z_coords.append(lm.z)

                    row = [CURRENT_LETTER] + x_coords + y_coords + z_coords

                    with open(OUTPUT_CSV, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    print(f"Captured sample for letter {CURRENT_LETTER}")
                else:
                    print("No hand detected, sample NOT captured.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
