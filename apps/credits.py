# -*- coding: utf-8 -*-
# @Author: prateek
# @Date:   2021-03-02 22:37:41
# @Last Modified by:   Prateek Agrawal
# @Last Modified time: 2022-04-20 10:32:43

import streamlit as st
def app():
    st.title(' Credits')
    st.write("""The following web application is built and maintained by **Group 1** as a course project for the course Digital Image Processing (CS5102)""")
    st.write("""Team Members : """)
    st.markdown("""
                1) B VIGNESH CED18I007 
                2) PRATEEK AGRAWAL CED18I040 
                3) THOTA SAI KEERTHANA CED18I053 
                4) UPPALAPATI PRANITA CED18I062 
                """)
    st.write("""

    ## Data
    The datasets consist of 5,863 high quality manually evalutated images of Chest X-Rays and consists of two categories **Pneumonia** and **Normal**.
    [Link to Dataset](https://drive.google.com/drive/folders/1aGjHxt5kODKb8STLm074sSkbNVRm7YIe?usp=sharing)
    """)
    
