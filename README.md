# Project Title

[AI Fridge Expiry Reminder]

## Project Description
[AI Smart Fridge is an AI-powered food expiry assistant. Users upload a food photo and expiry-date photo. The system recognizes the food, reads the date using OCR, predicts spoilage risk as Low, Medium, or High, creates reminders, saves items in My Fridge, and can send reminders to LINE.]

## Getting Started

[To run this project, users need Python and several required libraries. The project was mainly developed using Google Colab because it is easier to install libraries, test AI models, and run the Gradio web interface.
- Required Software
- Python
- Google Colab or local Python environment
- Gradio
- TensorFlow
- scikit-learn
- pandas
- numpy
- Pillow
- EasyOCR
- OpenCV
- Transformers
- PyTorch
- LINE Messaging API account, only if the LINE reminder feature is used
Installation
1. Install the required libraries:
-  !pip install gradio scikit-learn tensorflow pandas numpy matplotlib pillow easyocr opencv-python-headless transformers torch
-  If using the Kaggle food image dataset, also install KaggleHub:
-  !pip install kagglehub

2. Then download the food image dataset:
-  import kagglehub
-  dataset_path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
  
3. How to Use the Project
-  Open the notebook file AI_SMART_FRIDGE.ipynb.
-  Run all cells from top to bottom.
-  Wait until the Gradio interface appears.
-  Upload a food photo.
-  Upload an expiry-date label photo.
-  Enter the storage information:
-  Days stored
-  Package status
-  Storage location
-  Manual days until expiry, if OCR cannot read the date
-  Click the analyze button.
-  The system will show:
-  Recognized food name
-  OCR result
-  Days until expiry
-  Spoilage risk level
-  Recommendation
-  Reminder message
-  Save the item into My Fridge.
-  If LINE is connected, send the latest reminder to LINE.
  
4. Hugging Face Deployment
-  The project can also be deployed on Hugging Face Spaces using app.py and requirements.txt.
-  The public demo link is:
https://huggingface.co/spaces/Maxfh13/AIFRIDGEEXPIRYREMINDER
-  For LINE reminder deployment, the LINE Channel Access Token should be stored in Hugging Face Spaces under Variables and secrets as: LINE_CHANNEL_ACCESS_TOKEN
-  The token should not be written directly in the code or uploaded to GitHub.]

## File Structure
The project files are organized as follows:
AI-Smart-Fridge
1. AI_SMART_FRIDGE.ipynb
    - Main project notebook. This file contains the full development process,
    - Including library installation, dataset loading, food recognition,
    - OCR expiry-date reading, spoilage risk prediction, Gradio interface,
    - My Fridge dashboard, and LINE reminder function.

2. README.md
    - Project explanation, setup instructions, file structure, analysis,
    - results, contributors, acknowledgments, and references.
      
3. app.py
    - Python application file used for Hugging Face Spaces deployment.
    - It runs the Gradio interface online.
      
4. requirements.txt
    - List of Python libraries needed to run the Hugging Face app.

## Analysis
Analysis

This project uses AI and machine learning to solve the problem of forgotten food in the fridge. The main research question is:
How can AI help users know which food should be eaten first before it expires?
To answer this question, the project uses several analysis methods.
1. Food Recognition Analysis
- The first part of the system analyzes the food photo. The system uses image recognition to identify the food item. A CNN model is used for fruit and vegetable     recognition, while CLIP is used as a fallback for more general fridge foods such as yogurt, milk, bread, eggs, cheese, meat, and drinks. This makes the system     more useful because users do not always need to type the food name manually. They can upload a photo and let the AI recognize the food.

2. OCR Expiry-Date Analysis
- The second part of the system analyzes the expiry-date label photo. OCR, or Optical Character Recognition, is used to read printed text from the label.
  The OCR result helps the system find the expiry date and calculate how many days are left before the food expires. If the label is blurry, dark, or unclear, the   user can manually enter the number of days until expiry. This makes the system more reliable because it still works even when OCR fails.

3. Spoilage Risk Prediction
-  The third part of the system predicts the spoilage risk. A Random Forest model is used to classify the food into three risk levels:
   Low Risk
   Medium Risk
   High Risk
   The model uses these input features:
   Feature	        Purpose
   Food type	        Different foods have different shelf lives
   Days stored	    Food stored longer may have higher risk
   Package status	Opened food usually spoils faster
   Storage location	Fridge, freezer, and room temperature affect spoilage
   Days until expiry	Food closer to expiry has higher risk
-  The purpose of this analysis is not only to detect the expiry date, but also to understand the food situation. For example, opened milk stored for several days    may have a higher risk than unopened food with more days left before expiry.

4. Reminder Analysis
- After predicting the risk, the system creates a reminder message. The reminder tells the user whether the food is still safe, should be eaten soon, or should be   checked immediately. The item is saved into the My Fridge dashboard so users can track their food list. The system can also send the latest reminder to LINE.

5. Visualizations and Interface
-  The project uses the Gradio interface to present the analysis clearly. The interface includes:
   Food photo input
   Expiry-date label photo input
   Storage information form
   AI output result
   Risk level
   OCR result
   Recommendation
   My Fridge dashboard
   LINE reminder tab
-  The dashboard helps users quickly understand their fridge status, including total items, high-risk items, and food expiring soon.


TensorFlow Documentation.
https://www.tensorflow.org/

Kaggle Fruit and Vegetable Image Recognition Dataset.
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition
