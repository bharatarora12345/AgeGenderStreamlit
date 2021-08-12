import time

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import argparse
import time
import imutils
import pafy

# This is only for hide the water mark on stream lit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Face box method
def get_face_box(net, frame, conf_threshold=0.7):
    opencv_dnn_frame = frame.copy()
    frame_height = opencv_dnn_frame.shape[0]
    frame_width = opencv_dnn_frame.shape[1]
    blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [
        104, 117, 123], True, False)

    net.setInput(blob_img)
    detections = net.forward()
    b_boxes_detect = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            b_boxes_detect.append([x1, y1, x2, y2])
            cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2),
                          (0, 255, 0), int(round(frame_height / 150)), 8)
    return opencv_dnn_frame, b_boxes_detect
# Define paths of pretrained model
face_txt_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\opencv_face_detector.pbtxt'
face_model_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\opencv_face_detector_uint8.pb'

age_txt_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\age_deploy.prototxt'
age_model_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\age_net.caffemodel'

gender_txt_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\gender_deploy.prototxt'
gender_model_path = r'C:\Users\ACER\Desktop\Age_gender_prediction-main\Age_gender_prediction-main\gender_net.caffemodel'

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_classes = ['age: ~5', 'age: ~10', 'age: ~18', 'age: ~23', 'age: ~28', 'age: ~37', 'age: ~45', 'age: ~>50']
gender_classes = ['Male', 'Female']

age_net = cv2.dnn.readNet(age_model_path, age_txt_path)
gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path)
face_net = cv2.dnn.readNet(face_model_path, face_txt_path)


st.write("""
    #           Age and Gender predictor
    """)
select_Process = st.selectbox("Select process",("Browse Image" ,"Real time camera" , "Browse Videos"))

# First Process for browse image
if select_Process == "Browse Image":
    st.write("## Upload a picture that contains a face")
    uploaded_file = st.file_uploader("Choose a file:")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        cap = np.array(image)
        cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_RGB2BGR))
        cap = cv2.imread('temp.jpg')

        padding = 20
        t = time.time()
        frameFace, b_boxes = get_face_box(face_net, cap)
        if not b_boxes:
            st.write("No face Detected, Checking next frame")

        for bbox in b_boxes:
            face = cap[max(0, bbox[1] - padding):min(bbox[3] + padding, cap.shape[0] - 1),
                   max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            gender_pred_list = gender_net.forward()
            gender = gender_classes[gender_pred_list[0].argmax()]
            st.write(f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

            age_net.setInput(blob)
            age_pred_list = age_net.forward()
            age = age_classes[age_pred_list[0].argmax()]
            st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

            label = "{},{}".format(gender, age)
            cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            st.image(frameFace)
            st.markdown('**!!!!!!!!!  ThankYou For Using Our Age and Gender Predictor !!!!!!!!!!!**.')

# Second Process for Real time Camera
if select_Process == "Real time camera":
    start_Camera = st.selectbox("Select one : Start or Stop",("STOP" ,"START"));
    if(start_Camera == "START"):
        st.write("Processing...")
        parser = argparse.ArgumentParser()
        parser.add_argument('--image')
        parser.add_argument('-f')
        args = parser.parse_args()

        # # my_url='https://youtu.be/UbcrqCjgQgw'
        # my_url='https://youtu.be/dKd4sGfgdMQ'

        # vspapy=pafy.new(my_url)

        # playm=vspapy.getbest(preftype="mp4")
        video = cv2.VideoCapture(args.image if args.image else 0)

        # video=cv2.VideoCapture(playm.url)

        padding = 20

        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            # if not hasFrame:
            #     cv2.waitKey()
            #     break

            resultImg, faceBoxes = get_face_box(face_net, frame)
            if not faceBoxes:
                print("No face detected")

            for faceBox in faceBoxes:
                face = frame[max(0, faceBox[1] - padding):
                             min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding):
                                                                            min(faceBox[2] + padding, frame.shape[1] - 1)]

                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                gender_net.setInput(blob)
                genderPreds = gender_net.forward()
                gender = gender_classes[genderPreds[0].argmax()]
                print(f'Gender: {gender}')

                age_net.setInput(blob)
                agePreds = age_net.forward()
                age = age_classes[agePreds[0].argmax()]
                print(f'Age: {age[1:-1]} years')

                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                start_Camera == "STOP"
                break
        video.release()
        cv2.destroyAllWindows()
        st.write("**Camera Stopped....**")
        st.markdown('**!!!!!!!!!  ThankYou For Using Our Age and Gender Predictor !!!!!!!!!!!**.')
