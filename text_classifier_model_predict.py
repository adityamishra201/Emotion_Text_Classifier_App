# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:02:46 2023

@author: Admin
"""

import streamlit as st

import numpy as np
import pandas as pd

import joblib

pipe_lr = joblib.load(open("C:/Users/Admin/Desktop/Resume Project/emotion_classifier_pipe_lr.pkl","rb"))

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
    menu = ["Home","Monitor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
        st.subheader("Home-Emotion In Text")
        with st.form(key="emotion clear form"):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='submit')
        if submit_text:
            col1,col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            with col1:
                st.success("Original Text")
                st.write(raw_text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))

            with col2:
                st.success("Prediction Probability")
                st.write(probability)

        
    elif choice == "Monitor":
        st.subheader("Monitor App")
    
    else:
        st.subheader("About")





if __name__ == '__main__':
    main()