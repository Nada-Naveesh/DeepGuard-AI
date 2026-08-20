"""Reusable UI components."""

import streamlit as st


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">DeepGuard AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explainable Deepfake Image Detection System<br>'
        '<strong>AI-assisted deepfake image screening for educational and research use</strong></p>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown(
        """
DeepGuard AI is an academic deepfake image detection system designed to screen images
for potential AI-generated or manipulated content. The application uses the project's
existing TensorFlow/Keras deep-learning pipeline to process an uploaded image and
generate a predicted class with a confidence score.

The system is intended as an educational and research-oriented screening tool rather
than a definitive forensic verification system.
        """
    )
    st.markdown("---")


def render_sidebar():
    """Render sidebar with settings and return selected options."""
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 2rem 1.5rem; margin: -1rem -1rem 2rem -1rem;
                    border-radius: 0 0 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);'>
            <div style='text-align: center;'>
                <div style='background: rgba(255,255,255,0.2); width: 60px; height: 60px;
                           border-radius: 50%; margin: 0 auto 1rem auto; display: flex;
                           align-items: center; justify-content: center;
                           box-shadow: 0 4px 10px rgba(0,0,0,0.2);'>
                    <span style='font-size: 2rem;'>⚙️</span>
                </div>
                <h2 style='color: #ffffff; font-size: 1.8rem; font-weight: 800; margin: 0;
                           text-shadow: 0 2px 4px rgba(0,0,0,0.3); letter-spacing: 0.5px;
                           -webkit-font-smoothing: antialiased;'>
                    SETTINGS
                </h2>
                <p style='color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0.5rem 0 0 0;
                          font-weight: 500;'>
                    Configure Detection Parameters
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing Method Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-bottom: 1.5rem;
                    border: 2px solid rgba(255,255,255,0.2);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div style='display: flex; align-items: center; margin-bottom: 0.75rem;'>
                <div style='background: rgba(255,255,255,0.2); width: 35px; height: 35px;
                           border-radius: 8px; display: flex; align-items: center; justify-content: center;
                           margin-right: 0.75rem;'>
                    <span style='font-size: 1.2rem;'>📊</span>
                </div>
                <p style='color: #ffffff; font-weight: 700; margin: 0; font-size: 1.1rem;
                          text-shadow: 0 1px 2px rgba(0,0,0,0.2);'>
                    Preprocessing Method
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        **ℹ️ Overview of Preprocessing Methods**

        Input images undergo preprocessing transformations before model inference. The selected method significantly impacts detection accuracy.
        """)

        preprocessing_method = st.radio(
            "Select preprocessing method",
            options=[
                "✅ Training Match (Recommended)",
                "Simple [0,1] Normalization",
                "EfficientNet ImageNet"
            ],
            index=0,
            label_visibility="collapsed"
        )

        # Detailed explanation based on selection
        if "Training Match" in preprocessing_method:
            st.success("""
            **✅ Training Match (Recommended)**

            **Method Description:**
            - Employs identical preprocessing pipeline used during model training
            - BGR color format (OpenCV standard)
            - No normalization applied (raw pixel values: 0-255)

            **Rationale:**
            - Ensures consistency with training methodology
            - Provides optimal detection accuracy
            - Maintains full compatibility with trained model weights

            **✨ Recommended for conference demonstrations**
            """)
        elif "Simple" in preprocessing_method:
            st.warning("""
            **⚠️ Simple [0,1] Normalization**

            **Method Description:**
            - Processes images in RGB color format
            - Applies pixel-wise division by 255 (normalized range: 0-1)

            **Application Scenarios:**
            - Comparative analysis and ablation studies
            - Investigating normalization impact on detection performance

            **⚠️ Note:** Detection accuracy may be reduced, as this method differs from the training preprocessing pipeline.
            """)
        else:
            st.warning("""
            **📚 EfficientNet ImageNet Preprocessing**

            **Method Description:**
            - Applies ImageNet standardization protocol
            - Performs mean subtraction and variance scaling
            - Utilizes EfficientNet default preprocessing pipeline

            **Application Scenarios:**
            - Transfer learning applications with EfficientNet architecture
            - Compatibility with ImageNet pre-trained weights

            **⚠️ Note:** This model was trained using custom preprocessing. Performance may be suboptimal with this configuration.
            """)

        st.markdown("---")

        # Map selection to method string
        if "Training Match" in preprocessing_method:
            preprocess_method = "training_match"
        elif "Simple" in preprocessing_method:
            preprocess_method = "simple_norm"
        else:
            preprocess_method = "efficientnet"

        # Debug Mode Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-bottom: 1.5rem; margin-top: 1rem;
                    border: 2px solid rgba(255,255,255,0.2);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);'>
            <div style='display: flex; align-items: center; margin-bottom: 0.75rem;'>
                <div style='background: rgba(255,255,255,0.2); width: 35px; height: 35px;
                           border-radius: 8px; display: flex; align-items: center; justify-content: center;
                           margin-right: 0.75rem;'>
                    <span style='font-size: 1.2rem;'>🔧</span>
                </div>
                <p style='color: #ffffff; font-weight: 700; margin: 0; font-size: 1.1rem;
                          text-shadow: 0 1px 2px rgba(0,0,0,0.2);'>
                    Debug Mode
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("""
        **ℹ️ Debug Mode Overview**

        Displays detailed technical diagnostics for advanced analysis and system validation.
        """)

        show_debug = st.checkbox("Enable debug information", value=False)

        if show_debug:
            st.success("""
            **🔍 Debug Mode Enabled**

            **Diagnostic Information Displayed:**
            - 📷 Input image metadata (format, dimensions, color mode)
            - 🔄 Preprocessing pipeline details (tensor shape, data type, value range)
            - 📊 Raw model output (probability distributions)
            - 📈 Statistical metrics (mean, standard deviation)

            **Target Users:**
            - 👨‍💻 System developers
            - 🔬 Research personnel
            - 🎓 Technical demonstration audiences

            **Note:** Additional diagnostic information will appear in the analysis output section.
            """)
        else:
            st.info("""
            **💡 Standard Mode Active**

            Displays detection results only. Technical diagnostics are hidden.

            **To enable diagnostics:** Activate the checkbox above.
            """)

        # Pro Tip Section
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(255,215,0,0.15) 0%, rgba(255,193,7,0.1) 100%);
                    padding: 1.25rem; border-radius: 15px; margin-top: 2rem;
                    border: 2px solid rgba(255,215,0,0.3);
                    box-shadow: 0 4px 12px rgba(255,215,0,0.2);'>
            <div style='display: flex; align-items: flex-start;'>
                <div style='background: rgba(255,215,0,0.3); width: 40px; height: 40px;
                           border-radius: 50%; display: flex; align-items: center; justify-content: center;
                           margin-right: 1rem; flex-shrink: 0;'>
                    <span style='font-size: 1.4rem;'>💡</span>
                </div>
                <div>
                    <h4 style='color: #ffd700; font-weight: 700; margin: 0 0 0.5rem 0; font-size: 1rem;
                               text-shadow: 0 1px 2px rgba(0,0,0,0.3);'>
                        Recommended Configuration
                    </h4>
                    <p style='color: rgba(255,255,255,0.95); font-size: 0.85rem; margin: 0; line-height: 1.5;'>
                        The <strong style='color: #ffffff;'>Training Match</strong> preprocessing method matches the original training pipeline and is the recommended default.
                        Standard mode (debug disabled) provides a simpler academic presentation.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    return preprocess_method, show_debug


def render_footer():
    """Render the academic disclaimer and author footer."""
    st.markdown("---")
    st.warning(
        "This application is an educational screening tool. "
        "Predictions are probabilistic and may contain false positives and false negatives. "
        "It is not definitive forensic evidence."
    )
    st.markdown(
        """
<div style='text-align: center; padding: 1.5rem 1rem; background: #ffffff;
            border-radius: 12px; border: 1px solid #cbd5e0; margin-top: 0.5rem;'>
    <p style='margin: 0.25rem 0; color: #1a202c; font-weight: 700; font-size: 1.05rem;'>Nada Naveesh</p>
    <p style='margin: 0.25rem 0; color: #2d3748;'>Seshadri Rao Gudlavalleru Engineering College</p>
    <p style='margin: 0.25rem 0; color: #2d3748;'>B.Tech CSE – Artificial Intelligence and Machine Learning</p>
</div>
        """,
        unsafe_allow_html=True,
    )
