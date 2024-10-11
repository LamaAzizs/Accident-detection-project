import cv2
import streamlit as st
import pandas as pd
import os
import time
import tensorflow as tf  # Import TensorFlow for your custom model

# Load your TensorFlow classification model
model = tf.keras.models.load_model('models/my_model')

# Create "reports" and "report_images" folders if they don't exist
if not os.path.exists('reports'):
    os.makedirs('reports')
# Ensure 'temp' directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

if not os.path.exists('report_images'):
    os.makedirs('report_images')

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name, confidence, image_filename):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join('reports', report_filename)

    # Create a DataFrame for the report
    report_df = pd.DataFrame([{'Class': class_name, 'Confidence': confidence, 'Time': report_time}])

    # Save the DataFrame to a CSV file
    report_df.to_csv(report_path, index=False)

    # Add the report filename and associated image filename to the session state
    st.session_state.report_filenames.append((report_filename, image_filename))
    return report_filename

# Function to preprocess the video frame for TensorFlow model input
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Adjust based on your model's input size
    return resized_frame.reshape(1, 224, 224, 3)  # Reshape to match model's input shape

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
    accident_detected = False  # Track if an accident has been detected

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for the TensorFlow model
        preprocessed_frame = preprocess_frame(frame)

        # Predict using the TensorFlow model
        predictions = model.predict(preprocessed_frame)
        accident_prob = predictions[0][0]  # Assuming the first value is for 'accident' class

        # Convert the frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Choose the color based on accident probability
        if accident_prob > 0.9:
            color = (255, 0, 0)  # Red for high probability of accident
        else:
            color = (0, 255, 0)  # Green for low probability of accident

        # Add the accident probability percentage on the frame
        cv2.putText(frame_rgb, f'Accident likelihood: {accident_prob*100:.2f}%', 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Accident likelihood: {accident_prob*100:.2f}%")

        # Check if an accident is detected based on your threshold (e.g., > 0.5)
        if accident_prob > 0.9999 and not accident_detected:
            accident_detected = True  # Set the flag to true
            image_filename = f"image_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            image_path = os.path.join('report_images', image_filename)
            cv2.imwrite(image_path, frame)  # Save the image

            report_filename = generate_report("Accident", accident_prob, image_filename)
            st.write(f"Accident Detected: Confidence: {accident_prob*100:.2f}%. Report saved: {report_filename}")

        # Optional: Use time.sleep to slow down the frame display
        # time.sleep(frame_time)

    cap.release()
    st.write("Video playback complete.")

# Function to show reports
def show_reports():
    st.title("Accident Reports")

    # Sidebar to show report filenames and allow clicking to display report details
    st.header("Accident Reports")
    if st.session_state.report_filenames:
        for report_filename, image_filename in st.session_state.report_filenames:
            if st.button(report_filename):
                # Load and display the report details when clicked
                report_path = os.path.join('reports', report_filename)
                report_df = pd.read_csv(report_path)
                st.write(f"Details of {report_filename}:")
                st.dataframe(report_df)

                # Display the associated image
                image_path = os.path.join('report_images', image_filename)
                st.image(image_path, caption=f"Image for {report_filename}", use_column_width=True)
    else:
        st.write("No accident reports yet.")

# Streamlit app UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Video & Prediction", "Reports"])

if page == "Video & Prediction":
    st.title("Accident Classification Video Display")

    # File uploader for video input
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "gif"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_path = os.path.join('temp', 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())
        
        st.write(f"Uploaded video saved to: {video_path}")

        # Display the video with accident likelihood overlay
        display_annotated_video(video_path)

elif page == "Reports":
    show_reports()
