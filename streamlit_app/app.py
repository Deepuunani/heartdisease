import streamlit as st
import tensorflow as tf
import numpy as np
import sqlite3
import datetime
import tempfile
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from preprocessing.ecg_report_crop import crop_ecg_report
from preprocessing.extract_lead2 import extract_lead2
from preprocessing.image_to_signal import preprocess_image
from utils.ecg_info import DISEASE_INFO
from pdf_report import generate_pdf

from PIL import Image

st.set_page_config(
    page_title="AI ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)

st.title(
    "❤️ AI-Based ECG Arrhythmia Detection System"
)

st.subheader(
    "Early Heart Abnormality Screening"
)

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "saved_model/ecg_model.keras"
    )

model = load_model()

labels = {

    0: "Normal Beat",

    1: "Left Bundle Branch Block",

    2: "Right Bundle Branch Block",

    3: "Atrial Premature Beat",

    4: "Premature Ventricular Contraction",

    5: "Fusion Beat",

    6: "Paced Beat"

}

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
    "Upload ECG Report",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.image(
        image,
        caption="Uploaded ECG Report"
    )

    if st.button("Predict"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".png"
        ) as tmp:

            tmp.write(
                uploaded_file.getbuffer()
            )

            image_path = tmp.name

        cropped_path = crop_ecg_report(
            image_path
        )

        st.subheader(
            "Detected ECG Region"
        )

        st.image(
            Image.open(cropped_path),
            caption="Cropped ECG Region"
        )

        lead2_path = extract_lead2(
            cropped_path
        )

        st.subheader(
            "Detected Lead II Rhythm Strip"
        )

        st.image(
            Image.open(lead2_path),
            caption="Lead II Rhythm Strip"
        )

        sample = preprocess_image(
            lead2_path
        )

        prediction = model.predict(
            sample,
            verbose=0
        )

        predicted = np.argmax(
            prediction
        )

        confidence = float(
            np.max(prediction) * 100
        )

        disease = labels[predicted]

        if disease == "Normal Beat":

            status = "NORMAL"

        else:

            status = "ABNORMAL"

        st.success(
            "Prediction Completed"
        )
        st.markdown("---")

        st.header("Prediction Result")

        st.subheader("ECG Status")

        if status == "NORMAL":

            st.success("🟢 NORMAL")

        else:

            st.error("🔴 ABNORMAL")
        st.subheader("Diagnosis")

        if status == "NORMAL":
             st.success("No significant ECG abnormality detected.")
        else:
            st.error("Possible ECG abnormality detected. Further medical evaluation is recommended.")

        st.subheader("Confidence Score")

        st.progress(
            int(confidence)
        )

        st.write(
            f"**{confidence:.2f}%**"
        )

        info = DISEASE_INFO[
            disease
        ]

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
        st.subheader("Risk Suggestions")

        if status == "NORMAL":
            st.success("""
• Continue maintaining a healthy lifestyle.
• Exercise regularly.
• Follow a balanced diet.
• Get adequate sleep.
• Attend routine health check-ups.
""")

        else:
            st.warning("""
• Avoid strenuous physical activity until evaluated.
• Monitor symptoms regularly.
• Maintain a heart-healthy lifestyle.
• Avoid smoking and excessive alcohol.
• Seek medical advice if symptoms worsen.
""")
        st.subheader("Possible Symptoms")

        if status == "NORMAL":
            st.success("""
• No abnormal symptoms detected.
• Heart rhythm appears stable.
""")

        else:
            st.write("""
• Chest discomfort
• Palpitations
• Dizziness
• Fatigue
• Shortness of breath
""")

        
        st.subheader("Clinical Recommendation")

        if status == "NORMAL":
            st.success("""
Continue maintaining a healthy lifestyle.

• Exercise regularly.
• Eat a balanced diet.
• Get adequate sleep.
• Attend routine health check-ups.
""")

        else:
            st.warning("""
Consult a cardiologist for further evaluation.

Recommended next steps:
• Repeat ECG if advised.
• Additional tests such as Echocardiogram or Holter Monitoring may be required.
• Seek immediate medical attention if you experience chest pain, severe shortness of breath, fainting, or persistent palpitations.

Note: This AI prediction is intended for screening purposes only and should not replace a professional medical diagnosis.
""") 

        pdf_file = "ECG_Report.pdf"

        logo_path = "assets/mits_logo.png"

        generate_pdf(
            pdf_name=pdf_file,
            logo_path=logo_path,
            ecg_image_path=lead2_path,
            patient_name=name,
            age=age,
            gender=gender,
            status=status,
            disease=disease,
            confidence=confidence,
            info=info
        )

        with open(
            pdf_file,
            "rb"
        ) as pdf:

            st.download_button(
                label="📄 Download ECG Report",
                data=pdf,
                file_name="ECG_Report.pdf",
                mime="application/pdf"
            )

        connection = sqlite3.connect(
            "database/ecg_database.db"
        )

        cursor = connection.cursor()

        cursor.execute(
            """
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
            """
        )

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

        if os.path.exists(image_path):
            os.remove(image_path)

        if os.path.exists(cropped_path):
            os.remove(cropped_path)

        if os.path.exists(lead2_path):
            os.remove(lead2_path)

        st.success(
            "Prediction Saved Successfully"
        )

        st.markdown("---")

        st.success(
            "Analysis Completed Successfully"
        )

        st.balloons()