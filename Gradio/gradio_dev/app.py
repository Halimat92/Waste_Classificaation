import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import csv
import os
import requests

# # ---- Download Model from GitHub ----
# MODEL_URL = "https://github.com/Halimat92/Waste_Classificaation/blob/main/Gradio/gradio_dev/recycle_efficientnet_finetuned.keras"
# MODEL_FILE = "recycle_efficientnet_finetuned.kerasrecycle_model.keras"

# if not os.path.exists(MODEL_FILE):
#     with open(MODEL_FILE, "wb") as f:
#         f.write(requests.get(MODEL_URL).content)

# ---- Load Model ----
model = load_model("recycle_efficientnet_finetuned.keras")

# ---- Labels and Descriptions ----
class_names = ['batteries', 'clothes', 'e-waste', 'glass', 'light blubs', 'metal', 'organic', 'paper', 'plastic']

descriptions = {
    'batteries': 'Hazardous waste. Recycle at battery collection points.',
    'clothes': 'Reusable or recyclable textiles. Donate or use textile recycling bins.',
    'e-waste': 'Electronic waste. Contains toxic components. Use certified e-waste recyclers.',
    'glass': 'Non-biodegradable. Place in glass recycling containers.',
    'light blubs': 'May contain mercury. Dispose at special light bulb collection points.',
    'metal': 'Recyclable material. Sort into metal recycling containers.',
    'organic': 'Biodegradable. Compost or dispose in organic waste bins.',
    'paper': 'Recyclable if clean. Place in paper recycling bins.',
    'plastic': 'Non-biodegradable. Sort into plastic recycling bins.'
}

# ---- Sample Images ----
sample_images = [
    "sample_data/10537723_web1_M-Light-Bulb-EDH-180212.jpg",
    "sample_data/458-4583586_cardboard-recycling-png-paper-waste-clipart-transparent-png.png",
    "sample_data/assorted-clothes-isolated-heap-colorful-white-36145930.jpg",
    "sample_data/aa-batteries-energy-household-appliances-battery-recycling-used-alkaline-batteries-aa-size-format-207672475.jpg",
    "sample_data/banana-peel-white-background-composting-organic-waste-banana-peel-white-background-composting-organic-waste-210728827.jpg"
]

FEEDBACK_FILE = "user_feedback.csv"

# ---- Prediction Function ----
def predict(img):
    try:
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_label = class_names[predicted_index]
        confidence = float(np.max(predictions[0]))
        advice = descriptions[predicted_label]

        result_text = f"### {predicted_label.upper()} ({confidence*100:.2f}%)\n\n{advice}"
        if confidence < 0.50:
            result_text += "\n\n⚠️ Low confidence. Try a clearer image or better lighting."
        return result_text, predicted_label, confidence, advice
    except Exception:
        return "❌ No image detected. Please upload or take a photo.", "", 0.0, ""

# ---- Feedback Function ----
def save_feedback(label, confidence, advice, correct):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "predicted_label", "confidence_percentage", "advice", "correct_prediction"])
        writer.writerow([timestamp, label, f"{confidence*100:.2f}%", advice, correct])
    return "✅ Thanks for your feedback!"

# ---- Streamlit UI ----
st.title("♻️ Waste Classifier")
st.write("Upload or take a photo of waste to classify it and receive disposal guidance.")

uploaded_file = st.file_uploader("📸 Upload an Image", type=["jpg", "jpeg", "png"])
image_obj = None

if uploaded_file:
    image_obj = Image.open(uploaded_file)
    st.image(image_obj, caption="Uploaded Image", use_column_width=True)

if image_obj and st.button("🔍 Classify"):
    result, label, confidence, advice = predict(image_obj)
    st.markdown(result)

    st.write("### 📝 Was this prediction correct?")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Yes"):
            st.success(save_feedback(label, confidence, advice, "yes"))

    with col2:
        if st.button("👎 No"):
            st.success(save_feedback(label, confidence, advice, "no"))

# ---- Show Sample Images ----
st.markdown("### 🌍 Try with Sample Images")
cols = st.columns(len(sample_images))

for idx, img_path in enumerate(sample_images):
    with cols[idx]:
        st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)

st.info("**Note:** All predictions are currently in English. Multilingual support coming soon!")



