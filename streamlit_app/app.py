import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import datetime
import tempfile
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from preprocessing.image_to_signal import preprocess_image
from utils.ecg_info import DISEASE_INFO

# --------------------------------------------------

st.set_page_config(
    page_title="AI ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ AI-Based ECG Arrhythmia Detection System")

st.subheader(
    "Early Heart Abnormality Screening"
)

# --------------------------------------------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "saved_model/ecg_model.keras"
    )

model = load_model()

# --------------------------------------------------

labels = {

    0: "Normal Beat",

    1: "Left Bundle Branch Block",

    2: "Right Bundle Branch Block",

    3: "Atrial Premature Beat",

    4: "Premature Ventricular Contraction",

    5: "Fusion Beat",

    6: "Paced Beat"

}

# --------------------------------------------------

st.header("Patient Information")

name = st.text_input(
    "Patient Name"
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female",
        "Other"
    ]
)

uploaded_file = st.file_uploader(
    "Upload ECG Image",
    type=["png", "jpg", "jpeg"]
)
# --------------------------------------------------
# IMAGE PREVIEW AND PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
    image,
    caption="Uploaded ECG Image"
)

    if st.button("Predict"):

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())

            image_path = tmp.name

        # Convert uploaded image to ECG signal
        sample = preprocess_image(image_path)

        # Model Prediction
        prediction = model.predict(sample, verbose=0)

        predicted = np.argmax(prediction)

        confidence = float(
            np.max(prediction) * 100
        )

        disease = labels[predicted]
        if disease == "Normal Beat":
            status = "NORMAL"
        else:
            status = "ABNORMAL"

        st.success("Prediction Completed")

        # --------------------------------------------------
        # PREDICTION RESULT
        # --------------------------------------------------

        st.markdown("---")

        st.header("Prediction Result")

        st.subheader("ECG Status")

        if status == "NORMAL":
            st.success("🟢 NORMAL")

        else:
            st.error("🔴 ABNORMAL")

        st.subheader("Diagnosis")

        st.info(disease)

        st.subheader("Confidence Score")

        st.progress(int(confidence))

        st.write(
            f"**{confidence:.2f}%**"
        )

        # --------------------------------------------------
        # DISEASE INFORMATION
        # --------------------------------------------------

        info = DISEASE_INFO[disease]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Risk Level",
                info["Risk"]
            )

        with col2:

            st.metric(
                "Model Accuracy",
                "98.73%"
            )

        st.subheader("Description")

        st.info(
            info["Description"]
        )
        # --------------------------------------------------
        # POSSIBLE SYMPTOMS
        # --------------------------------------------------

        st.subheader("Possible Symptoms")

        if disease == "Normal Beat":

            st.success("No abnormal symptoms detected.")

        elif disease == "Premature Ventricular Contraction":

            st.write("""
• Palpitations
• Chest discomfort
• Dizziness
• Fatigue
""")

        elif disease == "Atrial Premature Beat":

            st.write("""
• Fast heartbeat
• Irregular heartbeat
• Mild dizziness
""")

        elif disease == "Left Bundle Branch Block":

            st.write("""
• Chest pain
• Shortness of breath
• Fatigue
""")

        elif disease == "Right Bundle Branch Block":

            st.write("""
• Usually no symptoms
• Mild dizziness
""")

        elif disease == "Fusion Beat":

            st.write("""
• Irregular heartbeat
• Weakness
• Fatigue
""")

        elif disease == "Paced Beat":

            st.write("""
• Pacemaker generated heartbeat
• Regular follow-up required
""")

        # --------------------------------------------------
        # CLINICAL RECOMMENDATION
        # --------------------------------------------------

        st.subheader("Clinical Recommendation")

        st.warning(
            info["Recommendation"]
        )

        # --------------------------------------------------
        # SAVE PREDICTION TO DATABASE
        # --------------------------------------------------

        connection = sqlite3.connect(
            "database/ecg_database.db"
        )

        cursor = connection.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            age INTEGER,
            gender TEXT,
            prediction TEXT,
            confidence REAL,
            date_time TEXT
        )
        """)

        cursor.execute(
            """
            INSERT INTO patients
            (
                patient_name,
                age,
                gender,
                prediction,
                confidence,
                date_time
            )
            VALUES
            (
                ?,?,?,?,?,?
            )
            """,
            (
                name,
                age,
                gender,
                disease,
                confidence,
                datetime.datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                )
            )
        )

        connection.commit()

        connection.close()

        # --------------------------------------------------
        # REMOVE TEMP IMAGE
        # --------------------------------------------------

        if os.path.exists(image_path):

            os.remove(image_path)

        # --------------------------------------------------
        # FINAL OUTPUT
        # --------------------------------------------------

        st.success(
            "Prediction Saved Successfully"
        )

        st.markdown("---")

        st.success(
            "Analysis Completed Successfully"
        )

        st.balloons()
        