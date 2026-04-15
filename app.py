import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd # Needed for the new bar chart
import cv2
import matplotlib.cm as cm

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 1. Safely handle inputs/outputs without extra brackets
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 2. Compute the Gradient
    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        last_conv_layer_output = outputs[0]
        preds = outputs[1]
        
        # TF Version safety: if preds is hidden inside a list, extract it
        if isinstance(preds, list):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Now we can safely slice it
        class_channel = preds[:, pred_index]

    # 3. Calculate weights and heatmap
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    
    # Bulletproof matrix multiplication syntax
    heatmap = last_conv_layer_output @ tf.expand_dims(pooled_grads, axis=-1)
    heatmap = tf.squeeze(heatmap)

    # 4. Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

    # Compute the Gradient
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate weights and heatmap
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224)) # Match your VGG16 input size

    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Colorize heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize and blend
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    
    # Convert BGR to RGB for Streamlit
    return cv2.cvtColor(np.uint8(superimposed_img), cv2.COLOR_BGR2RGB)



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
        image = ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color=(255, 255, 255))
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

            # --- GRAD-CAM EXPLAINABLE AI SECTION ---
            st.write("---")
            st.subheader("🧠 AI Attention Map (Grad-CAM)")
            st.write("This heatmap shows exactly where the AI looked to make its decision. Red/Yellow areas indicate high focus.")

            try:
                # 1. Save the uploaded file temporarily so OpenCV can read it
                temp_path = "temp_ecg.jpg"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())

                # 2. VGG16's last convolutional layer is usually named 'block5_conv3'
                last_conv_layer = "block5_conv3" 
                
                # 3. Generate the heatmap using your 'data' variable
                heatmap = make_gradcam_heatmap(data, model, last_conv_layer)
                
                # 4. Overlay it on the image
                gradcam_result = overlay_heatmap(temp_path, heatmap)

                # 5. Display Side-by-Side (Renamed columns to avoid conflict)
                gc_col1, gc_col2 = st.columns(2)
                with gc_col1:
                    st.image(file, caption="Original Input", use_container_width=True)
                with gc_col2:
                    st.image(gradcam_result, caption="AI Heatmap", use_container_width=True, channels="RGB")

            except Exception as e:
                st.warning(f"Could not generate Grad-CAM visualization. Error: {e}")