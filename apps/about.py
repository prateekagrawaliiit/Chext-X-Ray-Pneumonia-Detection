# -*- coding: utf-8 -*-
# @Author: prateek
# @Date:   2021-03-02 02:23:36
# @Last Modified by:   prateek
# @Last Modified time: 2021-03-02 23:04:21

import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from PIL import Image
def app():

	st.write("""    
    ## Problem Statement : 
    This Web Application attempts to identify whether the given Chest X-Ray Image is of a person having Pneumonia or of a Normal Person.
    """)
	st.write("""    
    ## Motivation : 
    Over the last few years with the advancement of technology and research in the field of machine learning and deep learning, they serve as a great resource and a tool that can be used in healthcare. Furthermore, owing to the current scenario the number of Chest related diseasses have grown exponentially. With the help of deep learning we attempt to serve as a base identifier for such chest related diseases that can help in early diagonosis and proper treatment.
    """)
	st.write(
        """
        ### Input : 
        
        The input is a gray scale image of a CHEST XRAY
        """)
	st.write(
        """
        ### Output : 
        Positive or Negative Label corresponding to pneumonia or normal chest xray.
        """)	
	st.write(
        """
        ### Processing : 
	To improve the quality of image we use Histogram Equalization on the gray scale image and then feed it to deep learning models.for predictions.

        """)
	st.write(
        """
        ### Models : 
	We use 3 different deep learning CNN models on two different varieties of images : Raw and Enhanced. The models namely are : AlexNet, VGG16_BN and ResNet50.
        """)
	st.write(
        """
        ### Conclusions : 
	Contrast Enhancement of Images yield much better accuracy and results than models trained on just raw chest X Ray images.
        """)


