import streamlit as st
import cv2
import tempfile
import pandas as pd
from backend.model_loader import load_trained_model
from backend.pipeline import run_full_pipeline

MODEL_PATH = r"Milestone_2\pcb_final_model.pth"

st.set_page_config(page_title="PCB Defect Detection", layout="wide")

st.title("🔍 PCB Defect Detection & Classification System")
st.markdown("Upload a template (defect-free) image and a test PCB image.")

# Load model once
@st.cache_resource
def load_model_once():
    return load_trained_model(MODEL_PATH)

model, class_names = load_model_once()

# Upload fields
col1, col2 = st.columns(2)

with col1:
    template_file = st.file_uploader(
        "Upload Template (Defect-Free PCB)",
        type=["jpg", "png", "jpeg"]
    )

with col2:
    test_file = st.file_uploader(
        "Upload Test PCB Image",
        type=["jpg", "png", "jpeg"]
    )

if template_file and test_file:

    # Save template temporarily
    temp_template = tempfile.NamedTemporaryFile(delete=False)
    temp_template.write(template_file.read())
    template_path = temp_template.name

    # Save test temporarily
    temp_test = tempfile.NamedTemporaryFile(delete=False)
    temp_test.write(test_file.read())
    test_path = temp_test.name

    st.info("Running full defect detection pipeline...")

    annotated_img, predictions = run_full_pipeline(
        test_path,
        template_path,
        model,
        class_names
    )

    if annotated_img is None:
        st.error("Processing failed.")
    else:

        colA, colB = st.columns([2,1])

        with colA:
            st.subheader("📌 Annotated PCB Output")
            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

        with colB:
            st.subheader("📊 Prediction Summary")

            if len(predictions) == 0:
                st.success("No defects detected 🎉")
            else:
                df = pd.DataFrame(predictions)
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Prediction Log (CSV)",
                    csv,
                    "pcb_predictions.csv",
                    "text/csv"
                )

        # Download annotated image
        output_path = "annotated_output.jpg"
        cv2.imwrite(output_path, annotated_img)

        with open(output_path, "rb") as file:
            st.download_button(
                "Download Annotated Image",
                file,
                "annotated_pcb.jpg",
                "image/jpeg"
            )