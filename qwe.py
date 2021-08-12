
import streamlit as st
from PIL import Image

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
import numpy as np


# In[2]:


# is it related to the read the data from the files 
import pandas as pd


# ###  .here we are importing another library pafy which is related to the 
# ###   url links ,import the video from the youtube.

# In[9]:




# In[11]:


import time
import imutils
import pafy


# #### .Import the  Open cv library

# In[12]:


# it is very useful library which is used to foucuses on the iamges ,processing,  captue and analyse the video
import cv2


# In[19]:



my_url='https://youtu.be/itIk2sNmiQo'

vspapy=pafy.new(my_url)

playm=vspapy.getbest(preftype="mp4")
st.write("""
    # Age and Gender prediction
    """)
image=Image.open(r'C:\Users\91971\Music\machine-learning-1920x1180.jpg')
st.image(image,use_column_width=True)
st.write("# Your Welcome...")
st.write("## press 0 for the ")

k=st.number_input("enter 0 for the your system webcam else 1")
st.write("the number you input  is",k)



# here i am just passing the link of youtube video.


capture_data = cv2.VideoCapture(0)

# here we are just manageing the size like length and width.

capture_data.set(3,480)

capture_data.set(4,640)

# here the mean value of the modal

mean_value= (78.4263377603, 87.7689143744, 114.895847746)

# here is the list of the age so we can predict 


list_of_age = ['age: ~5', 'age: ~9', 'age: ~15', 'age: ~20', 'age: ~25', 'age: ~35', 'age: ~45', 'age: ~>50']

# here is the list of the gender so we can predict the gender

list_of_gender = ['Gender: male', 'Gender: female']


def model_train():
    cal_age = cv2.dnn.readNetFromCaffe(r'C:\Users\91971\Documents\opencv_material\modelNweight_age_deploy.prototxt', r'C:\Users\91971\Documents\opencv_material\modelNweight_age_net.caffemodel')

    cal_gender = cv2.dnn.readNetFromCaffe(r'C:\Users\91971\Documents\opencv_material\modelNweight_gender_deploy.prototxt', r'C:\Users\91971\Documents\opencv_material\modelNweight_gender_net.caffemodel')
    return(cal_age, cal_gender)



def detection(cal_age, cal_gender):
    font=cv2.FONT_HERSHEY_SIMPLEX
    while True:
        # read the stream data 
        
        ret, image = capture_data.read()
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
         # cascade classifier we use to detect th face
        
        cascade_classifier = cv2.CascadeClassifier(r'C:\Users\91971\Documents\New folder_haar-classifier\haarcascade_frontalface_alt.xml')
        
        # here we just converting the image ito the gray image
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = cascade_classifier.detectMultiScale(gray, 1.3, 5)
        
        if(len(faces)>0):

            print("result  {} founded".format(str(len(faces))))
            
        for (x, y, w, h )in faces:
                
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            face_img = image[y:y+h, h:h+w].copy()
            
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), mean_value, swapRB=False)

            cal_gender.setInput(blob)
            
            gender_preds = cal_gender.forward()
            
                # predict the gender 
            
            gender_human = list_of_gender[gender_preds[0].argmax()]
            
                # print the type of thee gender
            
            print("the gender type: " + gender_human)


            cal_age.setInput(blob)
            
            age_preds = cal_age.forward()
            
                #predict the age
            
            age_human = list_of_age[age_preds[0].argmax()]
            
                # print the values of age
            
            print("the age range: " + age_human)

            overlay_text = "%s %s" % (gender_human, age_human)
            
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
                # show the final result   
        
        cv2.imshow('frame', image)  
        
                # this waityKey is used to  dispay a frame for the 1 ms 
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                # when you will enter the q it will stop the process of detection
            break
            
if __name__ == "__main__":
        
    # pass the valus in the functon
    
    cal_age, cal_gender = model_train()
    
    detection(cal_age, cal_gender)
    
capture_data.release()

cv2.destroyAllWindows()







# In[ ]:




