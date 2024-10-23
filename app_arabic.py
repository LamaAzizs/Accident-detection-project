import cv2
import streamlit as st
import pandas as pd
import os
import time
from ultralytics import YOLO
import numpy as np 
import google.generativeai as genai
from dotenv import load_dotenv
import random
from arabic_support import support_arabic_text

# Support Arabic text alignment in all components
support_arabic_text(all=True)
# Load the YOLO model
model = YOLO('runs/detect/train47/weights/best.pt')

# model = YOLO('bestyolo11.pt')
modelf = YOLO('injury.pt')

# model = YOLO('best(1).pt')

translation_dict = {
    'bike': 'دراجة',
    'car': 'سيارة',
    'injured': 'مصاب',
    'vehicle': 'مركبة'
}

# دالة لتعريب أسماء التصنيفات
def translate_class_name(class_name):
    # إذا كان الاسم يحتوي على كلمة "accident"، يُحوّل إلى "حادث"
    if 'accident' in class_name.lower():
        return 'حادث'
    
    # تقسيم الاسم الأصلي إلى كلمات وترجمة كل كلمة
    words = class_name.split()
    translated_words = [translation_dict.get(word.lower(), word) for word in words]
    translated_name = ' '.join(translated_words)
    return translated_name

# Create "reports" and "report_images" folders if they don't exist
if not os.path.exists('reports'):
    os.makedirs('reports')

if not os.path.exists('report_images'):
    os.makedirs('report_images')

# Initialize the session state for accident reports if it doesn't exist
if 'report_filenames' not in st.session_state:
    st.session_state.report_filenames = []

# Function to generate and save an accident report as CSV
def generate_report(class_name, image_filename, description=None):
    report_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())  # Current time for filename
    report_filename = f"report_{report_time}.csv"
    report_path = os.path.join('reports', report_filename)

    # تعريب اسم التصنيف
    translated_class_name = translate_class_name(class_name)

    # Create a DataFrame for the report, including a Description column
    report_data = {
        'Class': translated_class_name,  # استخدام الاسم المعرب هنا
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
# def generate_accident_report(image_path, yolo_report=None, time=None):
#     # Set up the API key
#     load_dotenv()
#     api_key = os.getenv('GOOGLE_API_KEY')
#     genai.configure(api_key=api_key)

#     # Upload the image using the File API
#     uploaded_file = genai.upload_file(path=image_path[-1], display_name="Accident Scene")
#     st.write(yolo_report)
#     # Construct the prompt
#     prompt = "Describe the accident based on the given image. make proficinal report without mentioning what you are doing \
#         make percise and consise so the officer can make decision be very sure whene makegin this report don't mention \
#             that this report is based on the image or it may not be accurate "
#     if yolo_report:
#         prompt += f" this is the calsses names for yolo report: {yolo_report}. Provide more details about the scene."
#     if time:
#         prompt += f" This incident occurred at {time}."

#     # Add specific instructions for the model to make the report useful for 911.
#     prompt += (" Include details such as the type and number of vehicles involved and there types and colors, "
#                "visible damage, any hazards, road conditions, presence of people, "
#                "and any blocked traffic. Provide suggestions for emergency response actions."
#                "i need it very short not more then 5 lines"
#                "and use spsific template like car1:\n car2:\n short descriptoin:\n and so on add any nedded failds")

#     # Choose the model and generate the response
#     try:
#         model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827")
#         response = model.generate_content([uploaded_file, prompt])
#         return response.text
#     except Exception as e:
#         return f"Failed to generate report: {str(e)}"

def generate_accident_report(image_path, yolo_report=None, time=None):
    # إعداد مفتاح API
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=api_key)

    # رفع الصورة باستخدام API
    uploaded_file = genai.upload_file(path=image_path[-1], display_name="صورة الحادث")

    # إعداد الطلب
    prompt = " قم بوصف الحادث بناءً على الصورة المقدمة. اجعل التقرير احترافيًا  مع عدم   ذكر أنه يعتمد على الصورة أ أو احتمالية عدم وجود حادث و   . "
    prompt += "يجب أن يكون التقرير منظمًا ويساعد المسؤول على اتخاذ القرار بسرعة. "
    if yolo_report:
        prompt += f"هذه هي أسماء الفئات وفقًا لتقرير YOLO: {yolo_report}. أضف المزيد من التفاصيل حول المشهد. "
    if time:
        prompt += f"وقع هذا الحادث في الساعة {time}. "

    # إضافة تعليمات محددة لجعل التقرير مفيدًا للطوارئ
    prompt += (
        "يجب أن يتضمن التقرير التفاصيل التالية بشكل منظم وكل عنوان من العناوين التالية  في سطر منفصل : "
        "نوع الحادث:"
        "المكان:"
        "المركبات:"
        " وصف الحادث:"
        "احذف الاسطر بين العناوين واذكر الاشياء الموجودة في وصف yolo لا تجعل العناوين كبيرة وقلل حجم التقرير"
        " ولا يوجد اسطر كثير, لا تتجاوز 8 اسطر مع وضع اسطر بعد كل فقرة بحيث تكون ويكون الوقت مع وصف الحادث لاتضع كل عنوان بالخط العريض "         
        )

    # اختيار النموذج وتوليد الرد
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro-exp-0827")
        response = model.generate_content([uploaded_file, prompt])
        return response.text
    except Exception as e:
        return f"فشل في توليد التقرير: {str(e)}"




def display_annotated_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    # Create a placeholder for the video frames
    frame_placeholder = st.empty()
    class_names = set()
    accident_detected = False
    list_of_images = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        new_width = 1080
        new_height = int(height * (new_width / width))
        frame_small = cv2.resize(frame, (new_width, new_height))
        # Annotate frame
        results = model.predict(source=frame_small, conf=0.6, device= 0,imgsz= 640)
       
        
        annotated_frame = results[0].plot()
        boxes_results = results[0].boxes
        boxes_resultsf = None
        
        if True:
            resultsf = modelf.predict(source=annotated_frame, conf=0.72, device= 0,imgsz= 640)
            annotated_frame = resultsf[0].plot() 
            boxes_resultsf = resultsf[0].boxes
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        else:
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
        
        if boxes_resultsf:
            for box in boxes_resultsf:
                class_id = int(box.cls[0])  # Class ID of the detected object
                class_name = modelf.names[class_id]
                translated_class_name = translate_class_name(class_name)
                if translated_class_name not in class_names:
                    st.info(f' تم اكتشاف {translated_class_name}', icon="ℹ️")
                    class_names.add(translated_class_name)

        for box in boxes_results:
            class_id = int(box.cls[0])  # Class ID of the detected object
            class_name = model.names[class_id]
            translated_class_name = translate_class_name(class_name)
            if translated_class_name not in class_names:
                st.info(f' تم اكتشاف {translated_class_name}', icon="ℹ️")
                class_names.add(translated_class_name)

            # Check if it's an accident in the current segment
            if "حادث" in translated_class_name:
                image_filename = f"image_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                image_path = os.path.join('report_images', image_filename)
                cv2.imwrite(image_path, frame)  # Save the image
                list_of_images.append(image_path)
                accident_detected = True

        

           

    if accident_detected:
        report_filename = generate_report(' '.join(class_names), list_of_images)
        st.toast(f"تم اكتشاف حادث: {' '.join(class_names)}. تم حفظ التقرير: {report_filename}", icon="🚨")

    cap.release()
    st.write("اكتمل تشغيل الفيديو.")

# Function to show reports
# Function to show reports
# Function to show reports
def show_reports():
    st.title("تقارير الحوادث")

    if st.session_state.report_filenames:
        for report_filename, image_paths in st.session_state.report_filenames:
            report_path = os.path.join('reports', report_filename)
            report_df = pd.read_csv(report_path)

            # عرض عنوان التقرير
            st.subheader(f"تقرير حول  ({','.join(report_df['Class'][0].split(' '))}) في {str(report_df['Time'][0])}")

            # عرض تفاصيل التقرير
            with st.expander("تفاصيل التقرير"):
                num_images_to_show = min(2, len(image_paths))
                selected_images = image_paths[0], image_paths[-1]

                # عرض الصور المختارة جنبًا إلى جنب
                cols = st.columns(num_images_to_show)
                for i in range(num_images_to_show):
                    cols[i].image(selected_images[i], use_column_width=True)

                # توليد وصف للتقرير إذا لم يكن موجودًا
                description_key = f"description_{report_filename}"
                if 'Description' in report_df.columns and pd.notnull(report_df['Description'][0]):
                    st.write("الوصف: \n", report_df['Description'][0])
                else:
                    if description_key not in st.session_state:
                        st.session_state[description_key] = ""
                    if st.button(f"توليد وصف لـ {report_df['Class'][0]}_{str(report_df['Time'][0])}"):
                        description = generate_accident_report(selected_images, f"{report_df['Class'][0]}", report_df["Time"][0])
                        st.session_state[description_key] = description
                        report_df['Description'] = description
                        report_df.to_csv(report_path, index=False)
                        st.success("تم توليد الوصف بنجاح!")

                    if st.session_state[description_key]:
                        st.write(st.session_state[description_key])
    else:
        st.write("لا توجد تقارير حوادث حتى الآن.")





# Streamlit app UI
st.sidebar.title("الواجهات")
page = st.sidebar.radio("الرجاء اختيار:", ["اكتشاف الحوادث من مقاطع الفيديو", "التقارير"])


if page == "اكتشاف الحوادث من مقاطع الفيديو":
    st.title("اكتشاف الحوادث من كاميرات المراقبة")
    uploaded_video = st.file_uploader("ارفع فيديو", type=["mp4", "avi", "mov", "gif"])

    if uploaded_video is not None:
        # Save uploaded video to a temporary file
        video_path = os.path.join('temp', 'uploaded_video.mp4')
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())

        # Display the annotated video using the placeholder
        display_annotated_video(video_path)


elif page == "التقارير":
    show_reports()
