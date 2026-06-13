

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

## Results
The final prototype successfully works as an AI-powered fridge expiry reminder system.
The main results are:
1. Food Recognition Result
-  The system can recognize food from an uploaded image. It can identify fruits, vegetables, and common fridge foods. This makes the system easier to use because     the user does not need to type the food name manually.

2. OCR Expiry-Date Reading Result
-  The system can read expiry-date labels using OCR. When the label photo is clear, the system can detect the expiry information. If the OCR result is not clear,     the manual input works as a backup.

3. Spoilage Risk Prediction Result
-  The system can predict spoilage risk as Low Risk, Medium Risk, or High Risk. The risk prediction is based on food type, days stored, package status, storage       location, and days until expiry. For example, an apple drink stored in the fridge with several days left before expiry was predicted as Low Risk. This means       the food still has low spoilage urgency and does not need immediate attention.

4. My Fridge Dashboard Result
-  The My Fridge dashboard successfully saves analyzed food items. It works like a fridge memory system, helping users remember what food they have and which         items need attention.
-  The dashboard can show:
   - Total food items
   - High-risk items
   - Items expiring soon
   - Food name
   - Risk level
   - Days left
   - Reminder message

5. LINE Reminder Result
-  The LINE reminder feature successfully sends the latest reminder to the user’s LINE account when the LINE Official Account, Messaging API, Channel Access          Token, and User ID are connected correctly. This makes the project more practical because users can receive reminders on LINE, an app many students already use    daily.

6. Online Demo Result
-  The project was deployed using Hugging Face Spaces. This means users can access the project through a public link without running Google Colab manually. This      is useful for exhibition testing because visitors can open the link or scan a QR code.
-  Overall, the result shows that AI Smart Fridge can help users manage food before it expires. The system is more useful than a normal reminder app because it       combines food recognition, OCR, risk prediction, dashboard tracking, and LINE notification.

## Contributor
- Name	                     Role
- Max Frederico Harmajie	   Project development, Gradio interface, AI system setup, LINE integration, Hugging Face deployment
- Kent Millian Peng          Project research, explanation, and testing support
- Valerie Keisha Ongkowijoyo Dataset organization, testing photos, and documentation support
- ivory Ameris               Presentation support, demo support, and final review

## Acknowledgments
We would like to thank our Introduction to AI course instructor for giving feedback and guidance during the project. The feedback helped us improve the project from a simple food reminder idea into a stronger AI system that includes food recognition, OCR, risk prediction, My Fridge dashboard, and LINE reminder integration.

We also acknowledge the open-source tools and platforms used in this project, including Python, Gradio, Hugging Face Spaces, EasyOCR, CLIP, TensorFlow, scikit-learn, and LINE Messaging API.

## References
Gradio Documentation.
https://www.gradio.app/

Hugging Face Spaces.
https://huggingface.co/spaces

LINE Messaging API Documentation.
https://developers.line.biz/en/docs/messaging-api/

EasyOCR GitHub Repository.
https://github.com/JaidedAI/EasyOCR

CLIP Model.
https://huggingface.co/openai/clip-vit-base-patch32



scikit-learn Random Forest Documentation.
https://scikit-learn.org/

