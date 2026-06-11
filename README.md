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
- !pip install gradio scikit-learn tensorflow pandas numpy matplotlib pillow easyocr opencv-python-headless transformers torch
- If using the Kaggle food image dataset, also install KaggleHub:
- !pip install kagglehub
2. Then download the food image dataset:
- import kagglehub
- dataset_path = kagglehub.dataset_download("kritikseth/fruit-and-vegetable-image-recognition")
3. How to Use the Project
- Open the notebook file AI_SMART_FRIDGE.ipynb.
- Run all cells from top to bottom.
- Wait until the Gradio interface appears.
- Upload a food photo.
- Upload an expiry-date label photo.
- Enter the storage information:
- Days stored
- Package status
- Storage location
- Manual days until expiry, if OCR cannot read the date
- Click the analyze button.
- The system will show:
- Recognized food name
- OCR result
- Days until expiry
- Spoilage risk level
- Recommendation
- Reminder message
- Save the item into My Fridge.
- If LINE is connected, send the latest reminder to LINE.
Hugging Face Deployment
- The project can also be deployed on Hugging Face Spaces using app.py and requirements.txt.
- The public demo link is:
https://huggingface.co/spaces/Maxfh13/AIFRIDGEEXPIRYREMINDER
- For LINE reminder deployment, the LINE Channel Access Token should be stored in Hugging Face Spaces under Variables and secrets as: LINE_CHANNEL_ACCESS_TOKEN
- The token should not be written directly in the code or uploaded to GitHub.]

## File Structure
The project files are organized as follows:

AI-Smart-Fridge
1. AI_SMART_FRIDGE.ipynb
    - Main project notebook. This file contains the full development process,
    - Including library installation, dataset loading, food recognition,
    - OCR expiry-date reading, spoilage risk prediction, Gradio interface,
    - My Fridge dashboard, and LINE reminder function.

2. app.py
    - Python application file used for Hugging Face Spaces deployment.
    - It runs the Gradio interface online.
3. requirements.txt
│   └── List of Python libraries needed to run the Hugging Face app.
│
4. README.md
│   └── Project explanation, setup instructions, file structure, analysis,
│       results, contributors, acknowledgments, and references.
│
6. images/
│   └── Screenshots of the interface, food scanning result, My Fridge dashboard,
│       and LINE reminder output.
│
7. docs/
│   └── Final paper, presentation slides, or exhibition notes.
│
8. fridge_inventory.csv
│   └── Generated file that stores saved food items in the My Fridge dashboard.
│
9. saved model files
    └── Food recognition or risk prediction model files, if saved after training.

The notebook is the main development file. The app.py file is used for the online demo. The requirements.txt file tells Hugging Face which libraries to install. The images folder stores visual proof of the project, while the docs folder stores written project materials.
