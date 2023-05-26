import cv2
import streamlit as st


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def main():
    # Create a video capture object for webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Check if the webcam is accessible
    if not cap.isOpened():
        st.error("Unable to access the webcam. Please make sure it is connected and try again.")
        return

    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier('C:/Users/Ameni/PycharmProjects/face recognition/haarcascade_frontalface_default.xml')

    # Create a sidebar in Streamlit for parameter adjustments
    st.sidebar.title('Parameter Adjustment')
    color_hex = st.sidebar.color_picker('Choose rectangle color', key="color_picker2", value='#FF0000')
    color_rgb = hex_to_rgb(color_hex)
    min_neighbors = st.sidebar.slider('minNeighbors', key="slider3", min_value=1, max_value=10, value=5)
    scale_factor = st.sidebar.slider('scaleFactor', key="slider4", min_value=1.1, max_value=2.0, value=1.3)

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        # Check if the frame was properly captured
        if not ret:
            st.error("Unable to capture frame from the webcam.")
            break

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color_rgb, 2)

        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Update parameter values based on sidebar inputs

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
