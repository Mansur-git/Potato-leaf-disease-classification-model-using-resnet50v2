import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

# 🖼️ Function to encode a local image to base64 (for background)
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# 📌 Encode background image (Ensure this image is in the same directory)
img_base64 = get_base64_image("potato-field-4357002_1280.jpg")

# 🎨 Custom CSS for Background & Styling
page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background: url("data:image/jpg;base64,{img_base64}") center/cover no-repeat;
    color: white;
}}
.result-box {{
    border-radius: 10px;
    padding: 15px;
    margin: 15px 0;
    font-size: 18px;
}}
.healthy {{ background-color: #00E676; color: white; }}
.early-blight {{ background-color: #FFC107; color: black; }}
.late-blight {{ background-color: #FF5252; color: white; }}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# 🏷️ App Title
st.markdown("<h1 style='text-align: center;'>🍃 Potato Leaf Disease Classifier 🍃</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Upload a potato leaf image to classify it.</h3>", unsafe_allow_html=True)

# 🔄 Load Model (Compatible with Streamlit 1.10)
@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model("potato_leaf_model.h5")

# 📤 File Upload Section
uploaded_file = st.file_uploader("📥 Upload a potato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # 🛠️ Preprocess Image
    image = image.convert("RGB")
    img_array = np.array(image.resize((256, 256))) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # 🔍 Predict Using Model
    try:
        model = load_model()
        with st.spinner("🔍 Analyzing the image..."):
            prediction = model.predict(img_array)

        class_names = ['Early Blight', 'Healthy', 'Late Blight']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # 🎯 Display Results
        if predicted_class == "Healthy":
            result_class = "healthy"
            message = f"✅ The leaf is **Healthy** ({confidence:.2f}%)"
            recommendation = "🌱 No pesticide needed. Keep monitoring your plants."
        elif predicted_class == "Early Blight":
            result_class = "early-blight"
            message = f"⚠️ Detected **Early Blight** ({confidence:.2f}%)"
            recommendation = "🛑 Apply a **moderate** amount of fungicide."
        else:  # Late Blight
            result_class = "late-blight"
            message = f"🚨 Detected **Late Blight** ({confidence:.2f}%)"
            recommendation = "🔥 Immediate action required! Apply **heavy fungicide**."

        st.markdown(f"""
            <div class="result-box {result_class}">
                <h2>{message}</h2>
                <h4>🌿 Recommendation: {recommendation}</h4>
            </div>
        """, unsafe_allow_html=True)

        # 📊 Show Confidence Scores
        st.markdown("### 🔍 Confidence Scores:")
        for i, class_name in enumerate(class_names):
            st.markdown(f"- **{class_name}**: `{prediction[0][i] * 100:.2f}%`")

    except Exception as e:
        st.error(f"❌ Model loading/prediction failed: {e}")
