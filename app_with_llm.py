import cv2
import streamlit as st
import pandas as pd
import os
import time
from ultralytics import YOLO
import numpy as np 
import google.generativeai as genai
from dotenv import load_dotenv


# Load the YOLO model
model = YOLO('runs/detect/train15/weights/best.pt')

# Create "reports" and "report_images" folders if they don't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

if not os.path.exists('report_images'):
    os.makedirs('report_images')

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name, confidence, image_filename, description=None):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join('reports', report_filename)

    # Create a DataFrame for the report, including a Description column
    report_data = {
        'Class': class_name,
        'Confidence': confidence,
        'Time': report_time,
        'Description': description  # New Description field
    }
    
    report_df = pd.DataFrame([report_data])

    # Save the DataFrame to a CSV file
    report_df.to_csv(report_path, index=False)

    # Add the report filename and associated image filename to the session state
    st.session_state.report_filenames.append((report_filename, image_filename))
    return report_filename


# Function to generate an accident description based on the image
def generate_accident_report(image_path, yolo_report=None, time=None):
    # Set up the API key
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)

    # Upload the image using the File API
    uploaded_file = genai.upload_file(path=image_path, display_name="Accident Scene")

    # Construct the prompt
    prompt = "Describe the accident based on the given image. make proficinal report without mentioning what you are doing \
        mmake percise and descriptive so the officer can make decision be very sure whene makegin this report don't mention \
            that this report is based on the image or it may not be accurate "
    if yolo_report:
        prompt += f" The system has detected: {yolo_report}. Provide more details about the scene."
    if time:
        prompt += f" This incident occurred at {time}."

    # Add specific instructions for the model to make the report useful for 911.
    prompt += (" Include details such as the type and number of vehicles involved and there types and colors, "
               "visible damage, any hazards, road conditions, presence of people, "
               "and any blocked traffic. Provide suggestions for emergency response actions.")

    # Choose the model and generate the response
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827")
        response = model.generate_content([uploaded_file, prompt])
        return response.text
    except Exception as e:
        return f"Failed to generate report: {str(e)}"

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

    # Get total frames and define segment size
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_size = total_frames // 3  # Split the video into 3 segments

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()
    detected_segment = False  # Track if an accident has been detected in the current segment

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Annotate frame
        height, width = frame.shape[:2]
        new_width = 640
        new_height = int(height * (new_width / width))
        frame = cv2.resize(frame, (new_width, new_height))

        results = model.predict(source=frame, conf=0.72)
        annotated_frame = results[0].plot()  # Annotated frame

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display the annotated frame in the placeholder
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        boxes = results[0].boxes  # Access detection boxes

        # Determine the current frame index
        current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        # st.write(current_frame_index, segment_size)
        # Reset the detected segment flag if moving to a new segment
        if current_frame_index % segment_size == 0 and current_frame_index > 0:
            detected_segment = False  # Reset the flag for the new segment

        # Check for accidents
        for box in boxes:
            class_id = int(box.cls[0])  # Class ID of the detected object
            confidence = box.conf[0]  # Confidence score
            class_name = model.names[class_id]  # Get class name

            # Check if it's an accident in the current segment
            if "accident" in class_name.lower() and not detected_segment:
                detected_segment = True  # Set the flag for the current segment
                image_filename = f"image_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = os.path.join('report_images', image_filename)
                cv2.imwrite(image_path, frame)  # Save the image

                # Generate report
                report_filename = generate_report(class_name, confidence, image_filename)
                st.write(f"Accident Detected: {class_name}, Confidence: {confidence:.2f}. Report saved: {report_filename}")

        # Wait before displaying the next frame
        time.sleep(frame_time)

    cap.release()
    st.write("Video playback complete.")

# Function to show reports
# Function to show reports
# Function to show reports
def show_reports():
    st.title("Accident Reports")

    if st.session_state.report_filenames:
        for report_filename, image_filename in st.session_state.report_filenames:
            report_path = os.path.join('reports', report_filename)
            report_df = pd.read_csv(report_path)

            # Display the report title
            st.subheader(f"{report_df['Class'][0]} in {str(report_df['Time'][0])}")

            # Display the associated image with a smaller size
            image_path = os.path.join('report_images', image_filename)
            st.image(image_path, caption=f"Image for {report_filename}", width=480)

            # Create a unique key for each report's description button
            description_key = f"description_{report_filename}"

            # Check if a description already exists
            

            # Display an expander to toggle the description visibility
            with st.expander("Report details"):
                description_exists = 'Description' in report_df.columns and pd.notnull(report_df['Description'][0])
                if description_exists:
                    st.write("**Description:**", report_df['Description'][0])
                else:
                    # Initialize the session state for description if it doesn't exist
                    if description_key not in st.session_state:
                        st.session_state[description_key] = ""

                    # Button for generating description
                    if st.button(f"Generate Description for {report_df['Class'][0]}_{str(report_df['Time'][0])}"):
                        # Generate a description for the report
                        description = generate_accident_report(
                            image_path, f"{report_df['Class'][0]} {report_df['Confidence'][0]}", report_df["Time"][0]
                        )
                        st.session_state[description_key] = description
                        
                        # Update the report DataFrame with the new description
                        report_df['Description'] = description
                        report_df.to_csv(report_path, index=False)
                        
                        st.success("Description generated successfully!")

                    # Display the description if it has been generated
                    if st.session_state[description_key]:
                        st.write("**Description:**", st.session_state[description_key])
    else:
        st.write("No accident reports yet.")





# Streamlit app UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Video & Prediction", "Reports"])

if page == "Video & Prediction":
    st.title("CCTV Accident Detection")

    # File uploader for video input
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "gif"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_path = os.path.join('temp', 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())
        
        # st.write(f"Uploaded video saved to: {video_path}")

        # Display the annotated video using the placeholder 
        display_annotated_video(video_path)

elif page == "Reports":
    show_reports()
