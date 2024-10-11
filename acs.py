import cv2
import streamlit as st
import pandas as pd
from ultralytics import YOLO
import os
import time

# Load the YOLO model
model = YOLO('trained_yolo_model.pt')

# Create "reports" folder if it doesn't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name, confidence):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join('reports', report_filename)

    # Create a DataFrame for the report
    report_df = pd.DataFrame([{'Class': class_name, 'Confidence': confidence, 'Time': report_time}])

    # Save the DataFrame to a CSV file
    report_df.to_csv(report_path, index=False)

    # Add the report filename to the session state
    st.session_state.report_filenames.append(report_filename)
    return report_filename

# Function to process video and display annotated frames
def display_annotated_video(video_path, fps_reduction=4):
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = max(1, original_fps // fps_reduction)  # Reduce FPS
    frame_time = 1.0 / new_fps  # Time to display each frame

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Annotate frame
        results = model.predict(source=frame, conf=0.850)
        annotated_frame = results[0].plot()  # Annotated frame

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame in the placeholder
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        boxes = results[0].boxes  # Access detection boxes

        # Construct log output and check for accidents
        for box in boxes:
            class_id = int(box.cls[0])  # Class ID of the detected object
            confidence = box.conf[0]  # Confidence score
            class_name = model.names[class_id]  # Get class name

            # Check if it's an accident
            if "accident" in class_name.lower():
                report_filename = generate_report(class_name, confidence)
                st.write(f"Accident Detected: {class_name}, Confidence: {confidence:.2f}. Report saved: {report_filename}")

        # Sleep to control the frame rate
        # time.sleep(frame_time)

    cap.release()
    st.write("Video playback complete.")

# Streamlit app UI
st.title("YOLO Annotated Video Display")

# Sidebar to show report filenames and allow clicking to display report details
with st.sidebar:
    st.header("Accident Reports")
    if st.session_state.report_filenames:
        for i, report_filename in enumerate(st.session_state.report_filenames):
            if st.button(report_filename, key=f"button_{i}"):
                # Load and display the report details when clicked
                report_path = os.path.join('reports', report_filename)
                report_df = pd.read_csv(report_path)
                st.write(f"Details of {report_filename}:")
                st.dataframe(report_df)
    else:
        st.write("No accident reports yet.")

# File uploader for video input
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    video_path = os.path.join('temp', 'uploaded_video.mp4')
    with open(video_path, 'wb') as f:
        f.write(uploaded_video.read())
    
    st.write(f"Uploaded video saved to: {video_path}")

    # Display the annotated video using the placeholder 
    display_annotated_video(video_path)
uploaded_video = None