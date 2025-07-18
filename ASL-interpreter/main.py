import cv2 
import mediapipe as m 

def main():
    #First to initialize the webcam
    cap = cv2.VideoCapture(0) # 0 -> default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    m_hands = m.solutions.hands
    m_drawings = m.solutions.drawing_utils

    with m_hands.Hands(
        static_image_mode = False,
        max_num_hands = 2,
        min_detection_confidence = 0.7,
        min_tracking_confidence = 0.7
    ) as hands:

        # Loop to continuously get frames from the webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Edit output frame
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    m_drawings.draw_landmarks(frame, hand_landmarks, m_hands.HAND_CONNECTIONS)

            # Display the frame
            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()