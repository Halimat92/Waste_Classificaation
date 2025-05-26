import gradio as gr
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from datetime import datetime
import csv
import os

# Load model
model = load_model("recycle_efficientnet_finetuned.keras")

# Class labels
class_names = ['batteries', 'clothes', 'e-waste', 'glass', 'light blubs', 'metal', 'organic', 'paper', 'plastic']

# Descriptions
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

# Sample images
sample_images = [
    "sample_data/10537723_web1_M-Light-Bulb-EDH-180212.jpg",
    "sample_data/458-4583586_cardboard-recycling-png-paper-waste-clipart-transparent-png.png",
    "sample_data/assorted-clothes-isolated-heap-colorful-white-36145930.jpg",
    "sample_data/aa-batteries-energy-household-appliances-battery-recycling-used-alkaline-batteries-aa-size-format-207672475.jpg",
    "sample_data/banana-peel-white-background-composting-organic-waste-banana-peel-white-background-composting-organic-waste-210728827.jpg"
    
]

# Feedback log file
FEEDBACK_FILE = "user_feedback.csv"

# Prediction logic
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

        result_text = f"{predicted_label.upper()} ({confidence*100:.2f}%)\n\n{advice}"
        if confidence < 0.50:
            result_text += "\n\n⚠️ Low confidence. Try a clearer image or better lighting."

        return result_text, predicted_label, confidence, advice
    except Exception:
        return "❌ No image detected. Please upload or take a photo.", "", 0.0, ""

# Save feedback
def save_feedback(label, confidence, advice, correct):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "predicted_label", "confidence_percentage", "advice", "correct_prediction"])
        writer.writerow([timestamp, label, f"{confidence*100:.2f}%", advice, correct])
    return "✅ Feedback saved. Thanks!"

# 🚫 No extra background styles here
with gr.Blocks() as demo:
    gr.Markdown("# ♻️ Waste Classifier")
    gr.Markdown("Upload or take a photo of waste to classify it and receive disposal guidance.")

    with gr.Row():
        img_input = gr.Image(
            label="📸 Take or Upload a Photo",
            type="pil",
            sources=["upload"],
            streaming=False
        )
        output_text = gr.Textbox(label="Prediction & Advice", lines=5)

    predicted_label_state = gr.State()
    confidence_state = gr.State()
    advice_state = gr.State()

    def handle_prediction(img):
        text, label, confidence, advice = predict(img)
        return text, label, confidence, advice

    btn = gr.Button("🔍 Classify")
    btn.click(
        fn=handle_prediction,
        inputs=img_input,
        outputs=[output_text, predicted_label_state, confidence_state, advice_state]
    )

    gr.Markdown("### 📝 Was this prediction correct?")
    feedback_status = gr.Textbox(label="", interactive=False)

    with gr.Row():
        yes_btn = gr.Button("👍 Yes")
        no_btn = gr.Button("👎 No")

    yes_btn.click(
        fn=lambda label, conf, adv: save_feedback(label, conf, adv, "yes"),
        inputs=[predicted_label_state, confidence_state, advice_state],
        outputs=feedback_status
    )

    no_btn.click(
        fn=lambda label, conf, adv: save_feedback(label, conf, adv, "no"),
        inputs=[predicted_label_state, confidence_state, advice_state],
        outputs=feedback_status
    )

    gr.Markdown("### 🌍 Try with Sample Images")
    gr.Gallery(
        value=sample_images,
        label="Sample Waste Images",
        columns=5,
        object_fit="contain"
    )

    gr.Markdown("**Note:** All predictions are currently in English. Multilingual support coming soon!")

demo.launch(share=True)

