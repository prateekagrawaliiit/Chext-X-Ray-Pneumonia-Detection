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
    The project aims to solve the problem of identifying whether the given image of CHEST XRAY has pneumonia or not.
    """)
    st.write("""    
    ## Motivation : 

    Over the last few years with the advancement of technology and research in the field of machine learning and deep learning, they serve as a great resource and a tool that can be used in healthcare. With the help of deep learning the predictions can serve as a serving stone for medical experts to build automated health monitoring systems and more.

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

    
    