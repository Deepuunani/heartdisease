from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors

import datetime
import os

styles = getSampleStyleSheet()
styles["Title"].alignment = 1
styles["Heading2"].spaceAfter = 8
styles["Normal"].spaceAfter = 6


def generate_pdf(
    pdf_name,
    logo_path,
    ecg_image_path,
    patient_name,
    age,
    gender,
    status,
    disease,
    confidence,
    info
):

    doc = SimpleDocTemplate(
        pdf_name,
        pagesize=A4
    )

    elements = []

    # ---------------------------------------
    # MITS HEADER
    # ---------------------------------------

    logo = Image(
        logo_path,
        width=0.9 * inch,
        height=0.9 * inch
    )

    heading = Paragraph(
        """
        <font size='18'><b>MADANAPALLE INSTITUTE OF TECHNOLOGY & SCIENCE</b></font><br/>
        <font size='12'>Department of Computer Science & Engineering (Data Science)</font><br/><br/>
        <font size='16'><b>AI-Based ECG Arrhythmia Detection Report</b></font>
        """,
        styles["Title"]
    )

    header = Table(
        [
            [
                logo,
                heading
            ]
        ],
        colWidths=[
            1.2 * inch,
            5.8 * inch
        ]
    )

    header.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 15)
            ]
        )
    )

    elements.append(header)

    elements.append(
        Spacer(
            1,
            20
        )
    )
    # ---------------------------------------
    # PATIENT INFORMATION
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>Patient Information</font></b>",

            styles["Heading2"]

        )

    )

    patient_table = Table(

        [

            ["Patient Name", patient_name],

            ["Age", str(age)],

            ["Gender", gender],

            [

                "Report Date",

                datetime.datetime.now().strftime(

                    "%d-%m-%Y %H:%M:%S"

                )

            ]

        ],

        colWidths=[2.2 * inch, 4.2 * inch]

    )

    patient_table.setStyle(

        TableStyle(

            [

                ("GRID", (0, 0), (-1, -1), 1, colors.black),

                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#D9EAF7")),

                ("BACKGROUND", (1, 0), (1, -1), colors.whitesmoke),

                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),

                ("FONTSIZE", (0, 0), (-1, -1), 11),

                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),

                ("TOPPADDING", (0, 0), (-1, -1), 8),

                ("ALIGN", (0, 0), (-1, -1), "LEFT")

            ]

        )

    )

    elements.append(patient_table)

    elements.append(

        Spacer(

            1,

            20

        )

    )    
    # ---------------------------------------
    # UPLOADED ECG IMAGE
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>Uploaded ECG Image</font></b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Spacer(

            1,

            10

        )

    )

    if os.path.exists(ecg_image_path):

        ecg = Image(

            ecg_image_path,

            width=6.2 * inch,

            height=2.5 * inch

        )

        ecg.hAlign = "CENTER"

        elements.append(ecg)

    else:

        elements.append(

            Paragraph(

                "<font color='red'>ECG Image Not Available</font>",

                styles["Normal"]

            )

        )

    elements.append(

        Spacer(

            1,

            20

        )

    )
    # ---------------------------------------
    # AI PREDICTION RESULT
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>AI Prediction Result</font></b>",

            styles["Heading2"]

        )

    )

    if status.upper() == "NORMAL":

        status_color = colors.green

    else:

        status_color = colors.red

    prediction_table = Table(

        [

            ["Status", status],

            ["Detected Disease", disease],

            ["Confidence", f"{confidence:.2f}%"],

            ["Risk Level", info["Risk"]]

        ],

        colWidths=[2.2 * inch, 4.2 * inch]

    )

    prediction_table.setStyle(

        TableStyle(

            [

                ("GRID", (0,0), (-1,-1), 1, colors.black),

                ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#D9EAF7")),

                ("BACKGROUND", (1,0), (1,-1), colors.whitesmoke),

                ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),

                ("FONTSIZE", (0,0), (-1,-1), 11),

                ("BOTTOMPADDING", (0,0), (-1,-1), 8),

                ("TOPPADDING", (0,0), (-1,-1), 8),

                ("TEXTCOLOR", (1,0), (1,0), status_color),

                ("ALIGN", (0,0), (-1,-1), "LEFT")

            ]

        )

    )

    elements.append(prediction_table)

    elements.append(

        Spacer(

            1,

            20

        )

    )
    # ---------------------------------------
    # DESCRIPTION
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>Disease Description</font></b>",

            styles["Heading2"]

        )

    )
 
    elements.append(
    Paragraph(
        info["Description"],
        styles["Normal"]
    )
)

    elements.append(

        Spacer(

            1,

            15

        )

    )

    # ---------------------------------------
    # POSSIBLE SYMPTOMS
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>Possible Symptoms</font></b>",

            styles["Heading2"]

        )

    )

    symptoms = []

    if disease == "Normal Beat":

        symptoms = [

            "No abnormal symptoms detected."

        ]

    elif disease == "Premature Ventricular Contraction":

        symptoms = [

            "Palpitations",

            "Chest discomfort",

            "Dizziness",

            "Fatigue"

        ]

    elif disease == "Atrial Premature Beat":

        symptoms = [

            "Fast heartbeat",

            "Irregular heartbeat",

            "Mild dizziness"

        ]

    elif disease == "Left Bundle Branch Block":

        symptoms = [

            "Chest pain",

            "Shortness of breath",

            "Fatigue"

        ]

    elif disease == "Right Bundle Branch Block":

        symptoms = [

            "Usually no symptoms",

            "Mild dizziness"

        ]

    elif disease == "Fusion Beat":

        symptoms = [

            "Irregular heartbeat",

            "Weakness",

            "Fatigue"

        ]

    elif disease == "Paced Beat":

        symptoms = [

            "Pacemaker generated heartbeat",

            "Regular follow-up required"

        ]

    for symptom in symptoms:

        elements.append(

            Paragraph(

                "• " + symptom,

                styles["Normal"]

            )

        )

    elements.append(

        Spacer(

            1,

            15

        )

    )

    # ---------------------------------------
    # CLINICAL RECOMMENDATION
    # ---------------------------------------

    elements.append(

        Paragraph(

            "<b><font size='14'>Clinical Recommendation</font></b>",

            styles["Heading2"]

        )

    )

    elements.append(

        Paragraph(

            info["Recommendation"],

            styles["Normal"]

        )

    )

    elements.append(

        Spacer(

            1,

            20

        )

    )
        
    elements.append(
        Paragraph(
            "<b>Generated By</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            """
            AI-Based ECG Arrhythmia Detection System<br/>
            Final Year Major Project<br/>
            Department of Computer Science & Engineering (Data Science)<br/>
            Madanapalle Institute of Technology & Science
            """,
            styles["Normal"]
        )
    )

    elements.append(
        Spacer(
            1,
            20
        )
    )

    doc.build(elements)  