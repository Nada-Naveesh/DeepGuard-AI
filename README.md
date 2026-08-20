# DeepGuard AI: Explainable Deepfake Image Detection System

**Author:** Nada Naveesh  
**College:** Seshadri Rao Gudlavalleru Engineering College  
**Course:** B.Tech CSE – Artificial Intelligence and Machine Learning

---

## Project Overview

DeepGuard AI is an academic interface for AI-assisted deepfake image screening. A user uploads a JPG, JPEG, or PNG image. The application validates the file, applies the repository’s existing preprocessing pipeline, and runs the existing TensorFlow/Keras model. The model returns a predicted class (`Fake` or `Real`) together with a probabilistic confidence score.

The system is intended for educational and research-oriented screening. It does not claim to detect every form of deepfake, and a single prediction should not be treated as forensic proof.

---

## Problem Statement

Generative models can produce images that are difficult to distinguish from authentic photographs. As synthetic imagery becomes more common, automated screening tools can help students and researchers inspect visual content. Automated systems remain imperfect: predictions are probabilistic, false positives and false negatives occur, and human or forensic review is still required before any authenticity decision is treated as conclusive.

---

## Objectives

1. Provide an accessible Streamlit interface for deepfake image screening.
2. Use the existing EfficientNetB7-based deep-learning model without replacing it.
3. Preserve the existing attention mechanism.
4. Process uploaded images through the existing preprocessing pipeline.
5. Display the predicted class and confidence score using the original label mapping.
6. Present technical information about architecture, input size, class, and confidence.
7. Provide a clear academic interface with validation, error handling, and a disclaimer.
8. Demonstrate the application of deep learning to image authenticity screening in an educational setting.

---

## Features

- Streamlit web interface
- JPG / JPEG / PNG upload
- Image validation for missing, unreadable, or corrupted files
- Existing TensorFlow/Keras inference pipeline
- EfficientNetB7 backbone
- Existing custom attention mechanism
- Existing preprocessing pipeline (training-match, simple normalization, EfficientNet ImageNet)
- Existing Hugging Face Hub model download (`CemRoot/deepfake-detection-model`)
- Predicted class using the original `Fake` / `Real` mapping
- Confidence score derived from the model softmax output
- Technical Details section
- User-facing error handling for invalid images and model-loading failure
- Academic disclaimer

---

## Methodology

The following diagram describes the repository’s existing inference path, verified against the source in `src/model/` and `src/preprocessing/`:

```text
Input Image
     ↓
Image Validation
     ↓
Image Preprocessing
     ↓
EfficientNetB7 Feature Extraction
     ↓
Attention Mechanism
     ↓
Classification Layer
     ↓
Prediction
     ↓
Class + Confidence Score
```

This diagram represents the existing inference architecture. It does not add training-time stages that are not executed by the Streamlit application.

### Inference steps in code

1. The uploaded file is opened with Pillow and rejected if it is not a readable image.
2. `preprocess_image()` resizes the image to **128 × 128**, which is the size used by `build_effatt_model()` and `DEFAULT_IMAGE_SIZE`.
3. The default preprocessing method (`training_match`) converts RGB to BGR and keeps float32 pixel values without `[0, 1]` normalization, matching the original training path documented in the source.
4. `load_model()` downloads `best_model_effatt.h5` from Hugging Face and loads it with Keras, including custom objects for the attention `RescaleGAP` layer. If full-model loading fails, the architecture is rebuilt and weights are loaded.
5. `model.predict()` produces a two-class softmax vector mapped as index `0 → Fake` and index `1 → Real`.
6. Confidence is the selected class probability expressed as a percentage. This is **model confidence for one image**, not dataset-level accuracy.

---

## Technologies Used

Verified from `requirements.txt` and source imports:

- Python
- Streamlit
- TensorFlow / Keras
- EfficientNetB7 (`tensorflow.keras.applications`)
- OpenCV (`opencv-python-headless`)
- Pillow
- NumPy
- Hugging Face Hub (`huggingface_hub`)

`packages.txt` lists system libraries used for Streamlit Cloud / Linux deployment (for example OpenCV-related packages such as `libgl1`). It is not a Python dependency file.

---

## Architecture

### Frontend

A Streamlit application. The entry point is `app.py`, which imports `main()` from `src/app.py`. UI styling and layout helpers are in `src/ui/`.

### Image Processing

Implemented in `src/preprocessing/image_processor.py`. Images of any resolution are resized to 128 × 128. Grayscale and RGBA inputs are converted to RGB before the selected preprocessing method is applied.

### Deep Learning Model

Defined in `src/model/architecture.py`: EfficientNetB7 (`include_top=False`) followed by batch normalization, a custom spatial attention block, dropout, a dense layer, and a two-unit softmax classifier.

### Model Loading

Implemented in `src/model/loader.py`. Weights are obtained from Hugging Face Hub repository `CemRoot/deepfake-detection-model`, file `best_model_effatt.h5`.

### Prediction

Classification uses the existing label mapping `['Fake', 'Real']`. The interface may describe Fake as potentially synthetic and Real as consistent with authentic content, but the class names themselves are unchanged.

---

## Dataset Description

According to the **original repository documentation**, the underlying research used a class-balanced collection of authentic and synthetic images, including GAN-based and diffusion-based generators. The original README reports a size of 20,000 images (10,000 real / 10,000 fake) and refers to a held-out test split, with further sourcing and licensing details placed in the published paper.

This academic adaptation does not independently verify those figures from raw dataset files inside this repository. No additional dataset name, participant count, or split statistics are claimed here beyond what the original documentation states.

If a local evaluation is required for this submission, it should be performed on a held-out set chosen by the student and recorded in the evaluation template below.

---

## Evaluation Metrics

Appropriate metrics for binary Fake/Real screening include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

**Model confidence** (the softmax probability shown in the app for one uploaded image) is not the same as **dataset-level evaluation accuracy**. A high confidence score on a single image does not mean the model is globally accurate.

Numerical accuracy, precision, recall, and F1 values for this academic adaptation are not independently measured in this repository. They need to be measured using a suitable held-out evaluation dataset. Figures reported in the original paper or original README should be cited as original-author results, not as results of this customization.

---

## Evaluation Results

Template for academic evaluation. Cells are left blank so that results are not fabricated.

| Expected Label | Predicted Label | Confidence | Correct/Incorrect |
|---|---|---:|---|
| Real |  |  |  |
| Fake |  |  |  |
| Real |  |  |  |
| Fake |  |  |  |

---

## Limitations

- Predictions are probabilistic.
- False positives can occur (authentic images predicted as Fake).
- False negatives can occur (manipulated images predicted as Real).
- Performance depends on the training and evaluation data used by the original model.
- Detection performance may vary across manipulation techniques and generators.
- Compression, resizing, and image quality may affect results.
- A model prediction should not be treated as definitive forensic evidence.
- The system should not be described as capable of detecting every deepfake.

---

## Future Scope

The following items are proposed future work. They are **not** implemented in this academic interface:

- Larger and more diverse datasets
- Stronger cross-dataset evaluation
- Improved explainability (for example localization of suspicious regions)
- Coverage of additional manipulation types
- Robustness testing under compression and post-processing
- More comprehensive benchmarking
- Video deepfake analysis
- Localization of manipulated regions
- Improved forensic analysis support
- Human-in-the-loop verification

---

## Academic Disclaimer

This application is an educational screening tool. Predictions are probabilistic and may contain false positives and false negatives. It is not definitive forensic evidence.

---

## Installation

### Requirements

- Windows, macOS, or Linux
- **Python 3.10, 3.11, or 3.12.** TensorFlow does not currently publish Windows wheels for Python 3.13/3.14, so `pip install tensorflow` fails on those versions (`No matching distribution found for tensorflow`). This machine should use `py -3.10`.
- Internet access on first run so the model can be downloaded from Hugging Face

### Windows (PowerShell)

```powershell
cd D:\deepfake-detection-streamlit

py -3.10 -m venv .venv

.venv\Scripts\activate

python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
cd deepfake-detection-streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### `packages.txt`

`packages.txt` is used by Streamlit Community Cloud (and similar Linux deploy environments) to install OS-level libraries required by OpenCV. It is not required for a typical local Windows `pip` install.

Python package versions in `requirements.txt` were left unchanged.

---

## Usage

1. Start the virtual environment.
2. Install dependencies from `requirements.txt` if they are not already installed.
3. Start Streamlit from the project root.
4. Open the local Streamlit URL (usually `http://localhost:8501`).
5. Upload a JPG, JPEG, or PNG image.
6. Wait for the model prediction (the model downloads from Hugging Face on first launch).
7. Review the predicted class (`Fake` or `Real`).
8. Review the confidence score.
9. Review the Technical Details section.

```powershell
cd D:\deepfake-detection-streamlit
.venv\Scripts\activate
streamlit run app.py
```

The application entry point is `app.py`.

Optional sidebar settings remain available: preprocessing method and debug diagnostics. **Training Match** is the method that matches the training preprocessing.

---

## Author

**Nada Naveesh**

Seshadri Rao Gudlavalleru Engineering College

B.Tech CSE – Artificial Intelligence and Machine Learning

Project: **DeepGuard AI: Explainable Deepfake Image Detection System**
