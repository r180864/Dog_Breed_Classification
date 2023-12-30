#importing Necessary Libraries

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from labels import class_names
from model import final_model as model

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Dog Breed Prediction')
st.text('Upload the Image')


input_shape=(375, 375)
model.load_weights("model.h5") #Loading Weights from pre trained Model

uploaded_file = st.file_uploader("Choose an Image...",type='JPG') #Accepting Image to upload
st.write("Upload Any Breed", class_names)
if uploaded_file is not None:
    img=Image.open(uploaded_file) #Opening the given Image
    st.image(img,caption='Uploaded Image')
    
    if st.button('PREDICT'):
        st.write('Result ...')
        img=img.resize(input_shape) #Resizing the image to our model accepted input shape
        st.image(img)
        test_image = image.img_to_array(img) #Converting the Image to Array
        test_image = np.expand_dims(test_image, axis = 0) #Expanding Dimentions from (375, 375, 3) --> (1, 375, 375, 3)
        prediction = model.predict(test_image,batch_size=32) ##Predicting the image using pre Trained DL Model
        st.write("Predicted label : ", class_names[np.argmax(prediction[0])]) #exporting max probability element/class/breed
        confidence = round(100 * (np.max(prediction[0])), 2) #Exporting the Probability the breed have
        st.write('Confidence : ',confidence)
