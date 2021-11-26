import numpy as np

import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import streamlit as st
import imageio
from PIL import Image

fav = Image.open("emblem_library.ico")
st.set_page_config(
    page_title="Book Cover Classifier",
    page_icon = fav,
)

st.title("Manning VS O'Reilly Classifier")


model = keras.models.load_model('xception_v1_03_0.983.h5')
#model = make_model(learning_rate=0.001, droprate=0.6, size=25)
#model.fit(train_ds, epochs=30, validation_data=val_ds, callbacks=[checkpoint])

uploaded_file = st.file_uploader("Upload Book Cover", type="jpg")
if uploaded_file is not None:
    uimg = Image.open(uploaded_file)
    uimg = uimg.save('379f996e73289292a42a9c2bc322a5d4.jpg')
    #st.image(uimg, caption='Uploaded Image', use_column_width=True)
    img_res=(100, 100)
    imgs = []
    img = imageio.imread('379f996e73289292a42a9c2bc322a5d4.jpg', pilmode = "RGB")
    img = np.array(Image.fromarray(img).resize(img_res))
    imgs.append(img)
    imgs = np.array(imgs)/127.5 - 1.
    imgs_A = imgs
    pred = model.predict(imgs_A)
    if pred[0]>0.5:
        st.write("That's an O'Reilly publication.")
    else:
        st.write("That's a Manning publication.")

