"""Entrypoint for the Student Performance Prediction app.

Run from the project root:

    streamlit run app.py

This file is the frame around every page: theme, sidebar, and navigation.
The page scripts in pages/ draw the main content.
"""

import streamlit as st

from src.ui import inject_css, render_sidebar

st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
render_sidebar()

navigation = st.navigation(
    [
        st.Page(
            "pages/1_Dashboard.py",
            title="Dashboard",
            icon=":material/space_dashboard:",
            default=True,
        ),
        st.Page(
            "pages/2_Predict_Performance.py",
            title="Predict Performance",
            icon=":material/calculate:",
        ),
        st.Page(
            "pages/3_Student_Records.py",
            title="Student Records",
            icon=":material/groups:",
        ),
        st.Page(
            "pages/4_Analytics.py",
            title="Analytics",
            icon=":material/monitoring:",
        ),
        st.Page(
            "pages/5_Model_Information.py",
            title="Model Information",
            icon=":material/model_training:",
        ),
    ],
    position="sidebar",
)
navigation.run()
