import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd # Needed for the new bar chart



# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CardioVision AI",
    page_icon="🫀",
    layout="wide", # Wider layout for a dashboard feel
    initial_sidebar_state="expanded"
)

# --- LOAD THE BRAIN (MODEL) ---
@st.cache_resource
def load_model():
    # Load the .keras model
    model = tf.keras.models.load_model('ecg_heart_model_v3.keras')
    return model

with st.spinner('Initializing AI System...'):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# --- SIDEBAR (PROJECT DETAILS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=100)
    st.title("CardioVision 🫀")
    st.subheader("Deep Learning for Cardiac Health")
    st.write("---")
    st.write("""
    **About:**
    This system uses a **VGG16 Convolutional Neural Network** to analyze ECG images and detect myocardial infarction (heart attacks).
    
    **Classes Detected:**
    - ✅ Normal
    - 🚨 Myocardial Infarction
    - ⚠️ History of MI
    - 💓 Abnormal Heartbeat
    """)
    st.write("---")
    st.caption("Built by Sahil Parvez | B.Tech CS Final Year")

# --- MAIN PAGE LAYOUT ---
st.title("🫀 CardioVision: Automated ECG Analysis")
st.write("Upload a paper-based ECG image to detect cardiac anomalies instantly.")

col1, col2 = st.columns([1, 1]) # Split screen into two columns

with col1:
    st.subheader("1. Upload ECG")
    file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Scan", use_container_width=True)

        # --- PREPROCESSING ---
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        normalized_image_array = (img_array.astype(np.float32) / 255.0)
        data = np.expand_dims(normalized_image_array, axis=0)

        # --- PREDICT BUTTON ---
        analyze_btn = st.button("🔍 Analyze Clinical Data", type="primary")

with col2:
    if file is not None and analyze_btn:
        st.subheader("2. Diagnosis Results")
        
        with st.spinner("Analyzing Signal Patterns..."):
            prediction = model.predict(data)
            
            # Get probabilities
            probs = prediction[0] 
            index = np.argmax(probs)
            confidence = probs[index]
            
            # Correct Labels
            class_names = [
                'Abnormal Heartbeat', 
                'History of MI', 
                'Myocardial Infarction', 
                'Normal'
            ]
            result_text = class_names[index]

            # --- DISPLAY METRICS ---
            # 1. Main Diagnosis
            if index == 3: # Normal
                st.success(f"✅ **{result_text}**")
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%", delta="Low Risk")
            elif index == 2: # MI (Heart Attack)
                st.error(f"🚨 **{result_text}**")
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%", delta="-CRITICAL", delta_color="inverse")
            else: # Others
                st.warning(f"⚠️ **{result_text}**")
                st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%", delta="Consult Doctor")

            # 2. Probability Chart (The "Data Science" Part)
            st.divider()
            st.write("### 📊 Probability Distribution")
            st.caption("How certain is the AI about other possibilities?")
            
            # Create a nice dataframe for the chart
            chart_data = pd.DataFrame({
                "Condition": class_names,
                "Probability": probs * 100
            })
            
            # Display Bar Chart
            st.bar_chart(chart_data, x="Condition", y="Probability", color="#FF4B4B")