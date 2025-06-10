An AI-powered, real-time computer vision system that helps users sort waste correctly at the point of disposal. This project supports smart recycling efforts by instantly classifying common waste types (e.g., plastic, paper, metal, food) using webcam input.
🎯 Objective
To improve recycling compliance and reduce contamination by providing instant, AI-driven waste classification through a public-facing, multilingual web interface.
🧩 Problem Statement
•	🚫 ~22% of recyclables are rejected due to improper sorting and contamination
•	🤷‍♂️ Users often lack clear disposal guidance at the moment of throwing waste
•	🏙️ Recycling standards vary across city regions, causing confusion
•	📢 Stakeholders need interactive demos to visualize AI-based waste solutions
🛠️ Solution Overview
A lightweight image classification model was fine-tuned and compared across EfficientNetB0 and MobileNetV2. EfficientNetB0 was selected for its superior performance and deployed via Hugging Face Spaces. A Gradio interface, integrated with OpenCV, captures live webcam input for real-time predictions, making the app both accessible and engaging.
🚀 Features
•	93% test accuracy and 93% macro F1-score across 9 waste categories
•	Real-time classification using webcam input
•	Clean, multilingual-ready Gradio UI
•	Deployed on Hugging Face for instant access no setup needed
