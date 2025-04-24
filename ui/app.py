import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import tempfile
import cv2
from core.crowd_counting_model import CrowdCountingModel

st.set_page_config(page_title="AI Video Analyzer", layout="centered")
st.title("Ru'ya!")

model_type = st.selectbox("Select Model", ["Crowd Counting", "Behavior Analysis"])

if "tracked_people" not in st.session_state:
    st.session_state.tracked_people = {}
if "next_id" not in st.session_state:
    st.session_state.next_id = 0
if "total_unique_people" not in st.session_state:
    st.session_state.total_unique_people = 0

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    ret, sample_frame = cap.read()
    cap.release()

    if ret:
        stframe = st.empty()

        if model_type == "Crowd Counting":
            model = CrowdCountingModel()
            frames = model.process(tfile.name, st.session_state)

        elif model_type == "Behavior Analysis":
            st.warning("Behavior Analysis model is not implemented yet.")
            frames = []

        for frame in frames:
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        st.session_state.tracked_people = {}
        st.session_state.next_id = 0
        st.session_state.total_unique_people = 0
