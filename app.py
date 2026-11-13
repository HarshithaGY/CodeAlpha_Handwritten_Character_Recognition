import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from streamlit_drawable_canvas import st_canvas


# ---------------------------------------------------------
# 1. PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="InkVision AI",
    page_icon="✍️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1020;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #11182d;
}
.main-title {
    color: #a78bfa;
    font-size: 40px;
    font-weight: 800;
}
.subtitle {
    color: #c4b5fd;
    font-size: 17px;
}
.metric-card {
    background: linear-gradient(135deg, #1e293b, #312e81);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #6366f1;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "inkvision_model.keras"


@st.cache_resource
def get_model(model_path):
    return tf.keras.models.load_model(model_path)


if not MODEL_PATH.is_file():
    st.error(
        "Model file not found. Put inkvision_model.keras "
        "in the same folder as app.py."
    )
    st.stop()

try:
    model = get_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ---------------------------------------------------------
# 3. IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(source_image):

    image = source_image.convert("L")

    # Convert black strokes on white background
    # to white strokes on black background.
    image = ImageOps.invert(image)

    # Resize to model input dimensions.
    image = image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Do not transpose: mouse drawings are already upright.

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image_array = image_array.reshape(
        1, 28, 28, 1
    )

    return image_array, image


# ---------------------------------------------------------
# 4. CHARACTER PREDICTION
# ---------------------------------------------------------

def make_prediction(image_array):

    output = model.predict(
        image_array,
        verbose=0
    )

    scores = np.asarray(output[0]).reshape(-1)

    if scores.size != 26:
        raise ValueError(
            f"Expected 26 classes, but model returned {scores.size}."
        )

    best_index = int(np.argmax(scores))

    best_letter = chr(
        ord("A") + best_index
    )

    confidence = float(
        scores[best_index]
    ) * 100

    return best_letter, confidence, scores


# ---------------------------------------------------------
# 5. DISPLAY PREDICTION
# ---------------------------------------------------------

def show_prediction(image_array, preview_image):

    letter, confidence, scores = make_prediction(
        image_array
    )

    st.markdown("---")
    st.subheader("AI Prediction")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <p>PREDICTED CHARACTER</p>
                <div style="font-size:90px;font-weight:800;color:#c4b5fd;">
                    {letter}
                </div>
                <h2>{confidence:.2f}%</h2>
                <p>Prediction Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.write("**Processed 28 × 28 Image**")

        st.image(
            preview_image.resize((224, 224)),
            width=224
        )

    st.subheader("Top 5 Predictions")

    top_five = np.argsort(scores)[::-1][:5]

    for index in top_five:

        name = chr(
            ord("A") + int(index)
        )

        value = float(scores[index])

        st.write(
            f"**{name}** — {value * 100:.2f}%"
        )

        st.progress(
            min(max(value, 0.0), 1.0)
        )


# ---------------------------------------------------------
# 6. SIDEBAR NAVIGATION
# ---------------------------------------------------------

st.sidebar.title("✍️ InkVision AI")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "✍️ Character Recognition",
        "📊 Model Analytics",
        "ℹ️ About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write("**Model:** CNN")
st.sidebar.write("**Dataset:** EMNIST Letters")
st.sidebar.write("**Classes:** A–Z")
st.sidebar.write("**Test Accuracy:** 93.04%")


# ---------------------------------------------------------
# 7. DASHBOARD
# ---------------------------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">✍️ InkVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Handwritten Character Recognition System</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Welcome to InkVision AI")

    st.write(
        "Draw one uppercase English letter or upload an image. "
        "The trained CNN will predict the character."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="metric-card">
            <h1>93.04%</h1>
            <p>Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="metric-card">
            <h1>26</h1>
            <p>Alphabet Classes</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="metric-card">
            <h1>CNN</h1>
            <p>Deep Learning Model</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Choose Character Recognition from the sidebar to try the model."
    )


# ---------------------------------------------------------
# 8. CHARACTER RECOGNITION
# ---------------------------------------------------------

elif page == "✍️ Character Recognition":

    st.markdown(
        '<div class="main-title">Character Recognition</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Draw a letter or upload a character image.</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    method = st.radio(
        "Choose Input Method",
        ["Draw Character", "Upload Image"],
        horizontal=True
    )

    # DRAWING MODE

    if method == "Draw Character":

        st.subheader("Draw One Uppercase Letter")

        st.caption(
            "Use your mouse to draw one letter in the white box."
        )

        canvas_col, help_col = st.columns([1, 1])

        with canvas_col:

            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0)",
                stroke_width=14,
                stroke_color="#000000",
                background_color="#FFFFFF",
                width=280,
                height=280,
                drawing_mode="freedraw",
                key="inkvision_drawing_canvas",
                
            )

            predict_clicked = st.button(
                "🔍 Predict Character",
                use_container_width=True
            )

        with help_col:

            st.subheader("How to Use")

            st.write("1. Draw one capital letter.")
            st.write("2. Draw near the centre.")
            st.write("3. Click Predict Character.")
            st.write("4. See the AI prediction.")

        if predict_clicked:

            if canvas_result.image_data is None:

                st.warning("Please draw a letter first.")

            else:

                try:

                    rgba = canvas_result.image_data.astype(
                        np.uint8
                    )

                    drawing = Image.fromarray(
                        rgba,
                        mode="RGBA"
                    ).convert("RGB")

                    gray = np.asarray(
                        drawing.convert("L")
                    )

                    if np.min(gray) > 245:

                        st.warning(
                            "The drawing board is blank. Draw a letter first."
                        )

                    else:

                        image_array, preview = preprocess_image(
                            drawing
                        )

                        show_prediction(
                            image_array,
                            preview
                        )

                except Exception as error:

                    st.error(
                        f"Prediction failed: {error}"
                    )

    # UPLOAD MODE

    else:

        st.subheader("Upload a Handwritten Character")

        uploaded = st.file_uploader(
            "Choose PNG, JPG or JPEG",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded is not None:

            try:

                uploaded_image = Image.open(
                    uploaded
                ).convert("RGB")

                st.image(
                    uploaded_image,
                    caption="Uploaded Image",
                    width=250
                )

                if st.button("🔍 Recognize Character"):

                    image_array, preview = preprocess_image(
                        uploaded_image
                    )

                    show_prediction(
                        image_array,
                        preview
                    )

            except Exception as error:

                st.error(
                    f"Could not process image: {error}"
                )


# ---------------------------------------------------------
# 9. MODEL ANALYTICS
# ---------------------------------------------------------

elif page == "📊 Model Analytics":

    st.markdown(
        '<div class="main-title">Model Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        "93.04%"
    )

    col2.metric(
        "Training Samples",
        "88,800"
    )

    col3.metric(
        "Testing Samples",
        "14,800"
    )

    st.subheader("CNN Architecture")

    architecture = [
        "Input: 28 × 28 × 1",
        "Convolutional Layer: 32 Filters",
        "Max Pooling",
        "Convolutional Layer: 64 Filters",
        "Max Pooling",
        "Convolutional Layer: 128 Filters",
        "Flatten",
        "Dense Layer: 128 Neurons",
        "Dropout: 30%",
        "Output: 26 Classes (A–Z)"
    ]

    for layer in architecture:

        st.write("• " + layer)

    st.subheader("Training Configuration")

    st.write("**Dataset:** EMNIST Letters")
    st.write("**Optimizer:** Adam")
    st.write("**Loss:** Sparse Categorical Crossentropy")
    st.write("**Output Activation:** Softmax")

    st.info(
        "93.04% is the reported test-set accuracy. "
        "Predictions on user drawings may differ."
    )


# ---------------------------------------------------------
# 10. ABOUT PROJECT
# ---------------------------------------------------------

elif page == "ℹ️ About Project":

    st.markdown(
        '<div class="main-title">About InkVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(
        "InkVision AI is a handwritten English character recognition "
        "application. It uses a Convolutional Neural Network trained on "
        "the EMNIST Letters dataset to classify letters from A to Z."
    )

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "TensorFlow and Keras",
        "Streamlit",
        "NumPy",
        "Pillow",
        "EMNIST Letters Dataset"
    ]

    for technology in technologies:

        st.write("• " + technology)

    st.subheader("Project Objective")

    st.write(
        "To demonstrate how deep learning and image preprocessing "
        "can be used to recognize handwritten alphabet characters."
    )


# ---------------------------------------------------------
# 11. FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;">
        ✍️ InkVision AI
        <br>
        Handwritten Character Recognition System
        <br>
        Powered by TensorFlow and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)