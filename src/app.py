"""
DeepGuard AI: Explainable Deepfake Image Detection System
Main Streamlit application.

Academic customization of the original CemRoot/deepfake-detection-streamlit
inference application. The TensorFlow/Keras prediction pipeline is unchanged.
"""

import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError

from model import load_model
from preprocessing import preprocess_image, DEFAULT_IMAGE_SIZE
from ui import apply_custom_css, render_header, render_footer, render_sidebar


st.set_page_config(
    page_title="DeepGuard AI | Explainable Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLASS_LABELS = ["Fake", "Real"]
ARCHITECTURE_LABEL = "EfficientNetB7 + Attention Mechanism"
INVALID_IMAGE_MESSAGE = (
    "Unable to process the uploaded image. Please upload a valid JPG, JPEG, or PNG image."
)


def open_uploaded_image(uploaded_file):
    """Validate that the upload is a readable image without altering preprocessing."""
    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file)
        image.load()
        return image
    except (UnidentifiedImageError, OSError, ValueError):
        st.error(INVALID_IMAGE_MESSAGE)
        return None


def run_prediction(model, image, preprocess_method, show_debug):
    """Run the existing preprocessing and Keras prediction pipeline."""
    img = preprocess_image(image, method=preprocess_method)
    if img is None:
        return None

    if show_debug:
        st.markdown("**Preprocessing Pipeline Diagnostics:**")
        st.write(f"- Selected Method: {preprocess_method}")
        st.write(f"- Tensor Shape: {img.shape}")
        st.write(f"- Data Type: {img.dtype}")
        st.write(f"- Value Range: [{img.min():.4f}, {img.max():.4f}]")
        st.write(f"- Statistical Mean: {img.mean():.4f}")
        st.write(f"- Standard Deviation: {img.std():.4f}")

    img_batch = np.expand_dims(img, axis=0)

    if show_debug:
        st.write(f"- Input Batch Shape: {img_batch.shape}")

    prediction = model.predict(img_batch, verbose=0)
    probs = np.array(prediction).squeeze().astype(float)

    if show_debug:
        st.markdown("**Model Output Diagnostics:**")
        st.write(f"- Output Tensor Shape: {prediction.shape}")
        st.write(f"- Probability Distribution: {probs}")

    if probs.ndim == 0 or len(probs) < 2:
        st.error("The model returned an unexpected output. No prediction was displayed.")
        return None

    answer_idx = int(np.argmax(probs))
    confidence = float(probs[answer_idx]) * 100
    pred_label = CLASS_LABELS[answer_idx]

    return {
        "pred_label": pred_label,
        "confidence": confidence,
        "fake_confidence": float(probs[0]) * 100,
        "real_confidence": float(probs[1]) * 100,
    }


def display_prediction(result):
    """Display predicted class and confidence using the existing label mapping."""
    pred_label = result["pred_label"]
    confidence = result["confidence"]

    st.markdown("### Prediction Result")
    st.markdown(f"**Predicted Class:** {pred_label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    if pred_label == "Fake":
        st.error("The image is predicted as **Fake** (potentially AI-generated or manipulated).")
    else:
        st.success("The image is predicted as **Real** (consistent with authentic photographic content).")

    st.markdown("**Classification Confidence Scores:**")
    st.markdown("Synthetic (Fake):")
    st.progress(min(max(result["fake_confidence"] / 100, 0.0), 1.0))
    st.write(f"{result['fake_confidence']:.2f}%")

    st.markdown("Authentic (Real):")
    st.progress(min(max(result["real_confidence"] / 100, 0.0), 1.0))
    st.write(f"{result['real_confidence']:.2f}%")


def display_technical_details(result=None):
    """Show verified architecture, input size, and the latest prediction if available."""
    input_w, input_h = DEFAULT_IMAGE_SIZE
    with st.expander("Technical Details", expanded=False):
        st.markdown("### Model Architecture")
        st.write(ARCHITECTURE_LABEL)

        st.markdown("### Input Image Size")
        st.write(f"Input Image Size: {input_w} × {input_h}")

        st.markdown("### Predicted Class")
        if result is None:
            st.write("Predicted Class: Not available until an image is analyzed.")
        else:
            st.write(f"Predicted Class: {result['pred_label']}")

        st.markdown("### Confidence Score")
        if result is None:
            st.write("Confidence Score: Not available until an image is analyzed.")
        else:
            st.write(f"Confidence Score: {result['confidence']:.2f}%")


def main():
    """Main application entry point."""
    apply_custom_css()
    render_header()

    model = load_model()
    if model is None:
        st.error(
            "The deep-learning model could not be loaded. "
            "Please verify the model configuration, Hugging Face connection, and required dependencies."
        )
        render_footer()
        return

    preprocess_method, show_debug = render_sidebar()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            "Select an image file...",
            type=["jpg", "jpeg", "png"],
            help="Upload a JPG, JPEG, or PNG image for AI-assisted deepfake screening.",
        )
        st.caption("Supported formats: JPG, JPEG, PNG")

        image = None
        file_id = None
        if uploaded_file is not None:
            file_id = f"{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("prediction_file_id") != file_id:
                st.session_state.pop("last_prediction", None)
                st.session_state["prediction_file_id"] = file_id
            image = open_uploaded_image(uploaded_file)
            if image is not None:
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if show_debug:
                    st.markdown("**Image Metadata:**")
                    st.write(f"- File Format: {image.format}")
                    st.write(f"- Color Mode: {image.mode}")
                    st.write(f"- Dimensions: {image.size}")

    with col2:
        st.markdown("### Analysis")

        if image is not None:
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    result = run_prediction(model, image, preprocess_method, show_debug)
                    if result is not None:
                        st.session_state["last_prediction"] = result

            result = st.session_state.get("last_prediction")
            if result is not None:
                display_prediction(result)
            else:
                st.info("Upload an image and select Analyze Image to view the predicted class and confidence.")
        else:
            st.info("Please upload a valid JPG, JPEG, or PNG image to begin screening.")
            st.session_state.pop("last_prediction", None)

    result = st.session_state.get("last_prediction")
    display_technical_details(result)
    render_footer()


if __name__ == "__main__":
    print("=" * 80)
    print("DeepGuard AI: Explainable Deepfake Image Detection System")
    print("=" * 80)
    main()
