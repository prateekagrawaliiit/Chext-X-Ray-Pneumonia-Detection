# -*- coding: utf-8 -*-
# @Author: prateek
# @Date:   2021-03-02 02:23:36
# @Last Modified by:   Prateek Agrawal
# @Last Modified time: 2022-04-20 15:20:28
import streamlit as st
import numpy as np
import joblib
import PIL
import os
import matplotlib.image as mpimg
from fastai.vision import open_image, load_learner, image, torch
from PIL import Image, ImageOps
from .predict_raw import infer
import torch
import time
from pathlib import Path
import cv2

def Image_Enhance(pth):
  img = cv2.imread(pth, 0)
  # print(img)
  h,w = img.shape
  freq = np.zeros(256).astype(np.int32)

  for i in range(h):
      for j in range(w):
          freq[img[i][j]] += 1
  for i in range(1,256):
    freq[i] += freq[i-1]
  freq = freq/(h*w)
  freq = freq*255
  freq = np.round(freq)
  img_new = np.zeros(img.shape)
  img_new.shape
  for i in range(h):
    for j in range(w):
        img_new[i][j] = freq[img[i][j]]
  return img_new



def app():
    st.title("Chest X-Ray Classification Application")
    st.header("Classification Example")
    option = st.radio('', ['Choose a Sample XRay', 'Upload your own XRay'])
    if option == 'Choose a Sample XRay':
        # Get a list of test images in the folder
        test_imgs = os.listdir("test_imgs_raw/")
        test_img = st.selectbox(
            'Please Select a Test Image:',
            test_imgs
        )
    # Display and then predict on that image
        fl_path = "test_imgs_raw/"+test_img
        img = open_image(fl_path)
        img_enhanced = Image_Enhance(fl_path)
        display_img = mpimg.imread(fl_path)
        st.image(display_img, caption="Chosen XRay", use_column_width=True)
        st.write("")
    predict = st.button("Classify the image")
    if predict:
        with st.spinner("Identifying the Disease..."):
            time.sleep(5)
        labels, probs, model_names = infer_raw(img)
        for i in range(len(labels)):
            st.success("Image Classified as : {} with a Confidence of : {2f} by using the model : {}".format(
                labels[i], probs[i], model_names[i]))
        # st.success(f"Image Disease: {label}, Confidence: {prob:.2f}%")

        with st.spinner("Identifying the Disease for enhanced image : "):
            time.sleep(5)
        labels, probs, model_names = infer_enhanced(img_enhanced)
        for i in range(len(labels)):
            st.success("Image Classified as : {} with a Confidence of : {2f} by using the model : {}".format(
                labels[i], probs[i], model_names[i]))

    st.warning("NOTE: If you upload an Image which is not a Chest XRay, the model will give very wierd predictions because it's trained to identify which one of the 2 labels the model is most confident of.")
    st.write("Project by Group : Group 1")
