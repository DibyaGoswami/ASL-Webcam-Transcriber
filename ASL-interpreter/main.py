import cv2 
import mediapipe as m 

def main():
    #First to initialize the webcam
    cap = cv2.VideoCapture(0) # 0 -> default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Loop to continuously get frames from the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Edit output frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (960, 720))

        # Display the frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()