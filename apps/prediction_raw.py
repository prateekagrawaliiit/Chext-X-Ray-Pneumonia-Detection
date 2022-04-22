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
from .predict_raw import infer_raw,infer_enhanced
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

  write_pth =  pth.split('.')[0] + '_enhanced.jpeg'
  cv2.imwrite(write_pth, img_new)
  return write_pth



def app():
    st.title("")
    st.header("Classification Example")
    st.write("Choose a Sample Image")
    test_imgs = os.listdir("test_imgs_raw/")
    test_img = st.selectbox(
            'Please Select a Test Image:',
            test_imgs
        )
    # Display and then predict on that image
    
    if test_img is not None: 
        fl_path = "test_imgs_raw/"+test_img
        img = open_image(fl_path)
        enhanced_img_pth = Image_Enhance(fl_path)
        img_enhanced = open_image(enhanced_img_pth)
        display_img = mpimg.imread(fl_path)
        st.image(display_img, caption="Raw XRay", use_column_width=True)
        st.write("Enhanced Image")
        display_img = mpimg.imread(enhanced_img_pth)
        st.image(display_img, caption="Enhanced Chest XRay", use_column_width=True)
        st.write("")
        predict = True
        if predict:
        	with st.spinner("Identifying the Disease..."):
        		time.sleep(5)
        		
        	labels, probs, model_names = infer_raw(img)
        	for i in range(len(labels)):
        		st.success("Image Classified as : {} with a Confidence of : {:.2f} by using the model : {}".format(labels[i], probs[i], model_names[i]))
        	with st.spinner("Identifying the Disease for enhanced image :"):
        		time.sleep(5)
        	st.write("""Predictions for Enhanced Image""")
        	labels, probs, model_names = infer_enhanced(img_enhanced)
        	for i in range(len(labels)):
        		st.success("Image Classified as : {} with a Confidence of : {:.2f} by using the model : {}".format(
		        labels[i], probs[i], model_names[i]))

    st.warning("NOTE: If you upload an Image which is not a Chest XRay, the model will give very wierd predictions because it's trained to identify which one of the 2 labels the model is most confident of.")
    st.write("Project by Group : Group 1")
