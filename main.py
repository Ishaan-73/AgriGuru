from flask import Flask, jsonify, request
from flask_cors import CORS
import torchvision.transforms as transforms 
import requests
import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from spellchecker import SpellChecker
from fuzzywuzzy import process
import numpy as np
from PIL import Image
import io
import base64
from torch import nn
import torch.nn.functional as F
import os
import torch
# import tensorflow as tf
torch.device('cpu')
#Model Class Implementation
class SimpleResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out) + x # ReLU can be applied before or after adding the input
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# base class for the model
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                   # Generate prediction
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {"val_loss": loss.detach(), "val_accuracy": acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x["val_loss"] for x in outputs]
        batch_accuracy = [x["val_accuracy"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))

def ConvBlock(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
             nn.BatchNorm2d(out_channels),
             nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(4))
    return nn.Sequential(*layers)


# resnet architecture 
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_diseases):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
        self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
        
        self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
        self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
        self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                       nn.Flatten(),
                                       nn.Linear(512, num_diseases))
        
    def forward(self, xb): # xb is the loaded batch
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out     

import pandas as pd

# nltk.download('punkt')
# nltk.download('wordnet')

import serial

app = Flask(__name__)
CORS(app)
data=""
data2=""


def wants_to_exit(user_input):
    exit_phrases = [
        "stop", "exit", "quit", "end", "no more", "that's all", "done", "finish", "no thanks", "goodbye", "bye"
    ]
    return any(phrase in user_input for phrase in exit_phrases)

def load_data(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_input(user_input):
    tokens = nltk.word_tokenize(user_input)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def correct_spelling(user_input):
    spell = SpellChecker()
    tokens = nltk.word_tokenize(user_input)
    corrected_tokens = [spell.correction(token) for token in tokens]
    return ' '.join(corrected_tokens)

def get_closest_match(user_input, data):
    scores = []
    processed_input = preprocess_input(user_input)
    for disease in data['diseases']:
        combined_text = disease['name'] + ' ' + ' '.join(disease['symptoms'])
        matching_words = set(processed_input.split()) & set(preprocess_input(combined_text).split())
        scores.append(len(matching_words))
    
    max_score = max(scores)
    if max_score == 0:
        return None
    
    best_match_index = scores.index(max_score)
    return data['diseases'][best_match_index]

def get_fuzzy_match(user_input, data):
    choices = [disease['name'] for disease in data['diseases']]
    best_match, score = process.extractOne(user_input, choices)
    
    if score < 80:  # You can adjust this threshold as needed
        return None
    
    for disease in data['diseases']:
        if disease['name'] == best_match:
            return disease

def list_diseases(data):
    disease_names = [disease['name'] for disease in data['diseases']]
    return ", ".join(disease_names[:-1]) + " and " + disease_names[-1]

def classify_intent(user_input):
    if any(word in user_input for word in ["prevent", "avoid"]):
        return "query_prevention"
    elif "symptom" in user_input:
        return "query_symptom"
    else:
        return "query_disease"

def is_greeting(user_input):
    greetings = ["hello", "hi", "hey", "greetings", "good day", "good morning", "good evening", "good afternoon"]
    return any(greet in user_input for greet in greetings)

def is_thank_you(user_input):
    thank_you_phrases = ["thank you", "thanks", "appreciate it", "grateful", "much appreciated"]
    return any(phrase in user_input for phrase in thank_you_phrases)

def chatbot(user_question):
    data = load_data('plant_diseases.json')
    responses = []
    responses.append("Greetings! Let's identify plant diseases. What symptoms have you observed or what would you like to know?")
    
    awaiting_feedback = False
    last_disease = None  # Store the last disease discussed for context
    
    while True:
        user_input = correct_spelling(user_question.strip().lower())
        
        # Split the user input based on conjunctions or punctuation
        user_segments = [seg.strip() for seg in user_input.split(" and ")]
        for segment in user_segments:
            feedback_handled = False  # Flag to check if feedback was handled

            if is_greeting(segment):
                positive_response="Hello! How can I assist you with plant diseases today?"
                responses.append(positive_response)
                continue

            if is_thank_you(segment):
                s="You're welcome! Let me know if there's anything else."
                responses.append(s)
                continue

            if "plant disease" in segment:
                responses.append("Here are some common plant diseases:")
                responses.append(list_diseases(data))
                continue

            if segment == 'help':
                s="Describe the symptoms you've observed in your plant, and I'll try to identify the potential disease. You can also ask questions related to plant diseases."
                responses.append(s)
                continue

            if wants_to_exit(segment):
                s="Goodbye! Take care."
                responses.append(s)
                break

            intent = classify_intent(segment)

            if intent == "query_prevention" and last_disease:
                s=f"To prevent {last_disease['name']}, it's a good idea to {last_disease['prevention'].lower()}."
                responses.append(s)
                continue

            disease = get_fuzzy_match(segment, data) or get_closest_match(segment, data)

            if disease:
                # Check if the user's input directly matches a disease name
                if disease['name'].lower() in segment:
                    s=f"You mentioned {disease['name']}. Here's what I know about it:"
                    responses.append(s)
                else:
                    responses.append(f"It sounds like your plant might be affected by {disease['name']}.")
                responses.append(f"{disease['name']} is characterized by {', '.join(disease['symptoms'][:-1])} and {disease['symptoms'][-1]}.")
                responses.append(f"To treat it, you can {disease['remedy'].lower()}. For prevention, it's a good idea to {disease['prevention'].lower()}.")
                prompt = "Was this information helpful? Or would you like to know about another disease or symptom? > "
                awaiting_feedback = True
                last_disease = disease  # Update the last disease discussed
                continue

            # More flexible feedback recognition
            positive_feedback_keywords = ['yes', 'yep', 'yeah', 'useful', 'helpful', 'informative','thanks','thank you']
            negative_feedback_keywords = ['no', 'nah', 'not really', 'unhelpful']

            if any(keyword in segment for keyword in positive_feedback_keywords) and awaiting_feedback:
                s="I'm glad to hear that! Let me know if there's anything else."
                responses.append(s)
                awaiting_feedback = False
                feedback_handled = True

            elif any(keyword in segment for keyword in negative_feedback_keywords) and awaiting_feedback:
                s="I'm sorry to hear that. Let's try again. Please provide more details or ask another question."
                responses.append(s)
                awaiting_feedback = False
                feedback_handled = True

            # If feedback was handled, skip the rest of the processing for this segment
            if feedback_handled:
                continue

            responses.append("I couldn't quite identify the disease from the details provided. Could you describe the symptoms more or ask about a specific disease?")
        print(responses[-1])
        return responses[-1]



# model = tf.keras.models.load_model("AlexNetModel.hdf5")
num_diseases = 38
in_channels = 3  

model = ResNet9(in_channels=in_channels, num_diseases=num_diseases)
PATH = "./resnet.pth"
model.load_state_dict(torch.load(PATH,map_location='cpu'))
model.eval()


# import joblib
# model2 = joblib.load('RandomForest.pkl')

import requests
import re
API_KEY = 'AIzaSyA-AEOb1UEZKyBndGETZa844Vy3Jl7wChM'
SEARCH_ENGINE_ID = 'c261acd80ca1f40d9'


li = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
def predict_class_from_base64_image(base64_image, model, class_list):
    try:
        # Remove the data URL prefix if it exists (e.g., "data:image/png;base64,")
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]

        # Decode the base64 image to bytes
        image_bytes = base64.b64decode(base64_image)

        # Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert grayscale to RGB (repeat single channel 3 times if image is grayscale)
        if image.mode != 'RGB':
            image = image.convert("RGB")

        # Apply the ToTensor() transformation
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize image to 256x256
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)  # Ensure 3 channels
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension [1, 3, H, W]

        # Make predictions using the PyTorch model
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            output = model(image_tensor)  # Forward pass
            _, predicted_class_index = torch.max(output, 1)  # Get the index of the highest logit

        # Convert predicted index to class label
        predicted_class = class_list[predicted_class_index.item()]

        return predicted_class

    except Exception as e:
        return str(e)
# Example usage:
image_input_base64 = ""
predicted_class = predict_class_from_base64_image(image_input_base64, model, li)
print("Predicted Class:", predicted_class)




def get_plant_cure(disease_name):
    # Define a dictionary with disease names and their corresponding cures.
    disease_cures = {
        'Apple___Apple_scab': 'Apply fungicide spray to affected areas.',
        'Apple___Black_rot': 'Prune and remove affected branches. Apply fungicide.',
        'Apple___Cedar_apple_rust': 'Apply fungicide in spring to prevent infection.',
        'Apple___healthy': 'Here is a fun fact about apple trees: Apples are a great source of dietary fiber!',
        'Blueberry___healthy': 'Blueberries are packed with antioxidants and are a healthy snack!',
        'Cherry_(including_sour)___Powdery_mildew': 'Apply a fungicide to control powdery mildew.',
        'Cherry_(including_sour)___healthy': 'Did you know cherries are a natural source of melatonin, which may help with sleep?',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Use fungicides and practice crop rotation.',
        'Corn_(maize)___Common_rust_': 'Plant resistant corn varieties if available.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Plant resistant corn varieties and consider fungicide treatment.',
        'Corn_(maize)___healthy': 'Corn is a staple food in many cultures and is a good source of carbohydrates.',
        'Grape___Black_rot': 'Prune and destroy affected grape clusters. Apply fungicides.',
        'Grape___Esca_(Black_Measles)': 'Prune and destroy affected vines. Use fungicides.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Prune and remove infected leaves. Apply fungicides.',
        'Grape___healthy': 'Grapes are not only delicious but are also used to make wine and raisins!',
        'Orange___Haunglongbing_(Citrus_greening)': 'There is no known cure for this disease. Remove and destroy infected trees.',
        'Peach___Bacterial_spot': 'Use copper-based sprays and practice sanitation.',
        'Peach___healthy': 'Peaches are known for their sweet and juicy flavor.',
        'Pepper,_bell___Bacterial_spot': 'Use copper-based sprays and avoid overhead irrigation.',
        'Pepper,_bell___healthy': 'Bell peppers are a great source of vitamin C and add flavor to many dishes.',
        'Potato___Early_blight': 'Remove infected leaves and use fungicides.',
        'Potato___Late_blight': 'Remove infected leaves and use fungicides. Practice crop rotation.',
        'Potato___healthy': 'Potatoes are a versatile food and can be prepared in various ways.',
        'Raspberry___healthy': 'Raspberries are rich in vitamins and antioxidants and can be enjoyed fresh or in desserts.',
        'Soybean___healthy': 'Soybeans are a valuable source of protein and are used in many food products.',
        'Squash___Powdery_mildew': 'Apply fungicides and practice good garden hygiene.',
        'Strawberry___Leaf_scorch': 'Remove infected leaves and use fungicides.',
        'Strawberry___healthy': 'Strawberries are a delicious and nutritious fruit that can be eaten fresh or used in various dishes.',
        'Tomato___Bacterial_spot': 'Remove infected leaves and use copper-based sprays.',
        'Tomato___Early_blight': 'Remove infected leaves and use fungicides.',
        'Tomato___Late_blight': 'Remove infected leaves and use fungicides. Avoid overhead irrigation.',
        'Tomato___Leaf_Mold': 'Use fungicides and ensure good air circulation.',
        'Tomato___Septoria_leaf_spot': 'Remove infected leaves and use fungicides.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Use insecticidal soap or neem oil to control spider mites.',
        'Tomato___Target_Spot': 'Remove infected leaves and use fungicides.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'There is no known cure for this virus. Use insecticides to control the vector.',
        'Tomato___Tomato_mosaic_virus': 'There is no known cure for this virus. Remove and destroy infected plants.',
        'Tomato___healthy': 'Tomatoes are a versatile ingredient in cooking and are packed with vitamins and minerals.'
    }

    # Check if the provided disease name is in the dictionary.
    if disease_name in disease_cures:
        return disease_cures[disease_name]
    else:
        # If the plant is healthy or the disease is not in the list, provide a generic message.
        if 'healthy' in disease_name.lower():
            return 'Here is a fun fact about this plant: It is healthy and thriving!'
        else:
            return 'No specific cure information available for this disease.'

# UPLOAD_FOLDER = 'uploaded_images'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and allowed_file(file.filename):
#         filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filename)
#         return jsonify({"success": True, "message": "File uploaded!"}), 200

#     return jsonify({"error": "Invalid file type"}), 400

def func3(location):
    search_query=f'Monthly Rainfall in mm in {location}'
    print(search_query)
    url = 'https://www.googleapis.com/customsearch/v1'

    params={
        'q':search_query,
        'key':API_KEY,
        'cx': SEARCH_ENGINE_ID
    }
    response = requests.get(url,params=params)
    results = response.json()
    print(results['items'][0]['snippet'])
    for i in range(0,len(results['items'])):
        text =results['items'][i]['snippet']
        # Regular expression pattern to match the first value before 'mm' or ' mm'
        pattern = r'(\d+\.\d+|\d+)\s*mm'
        # Find all matches using the regular expression pattern
        matches = re.findall(pattern, text)
        #Remove ',' from matches[0]
        if matches:
            # matches[0]=re.sub(',','',matches[0])
            rf_mm = float(matches[0])
            if rf_mm>300:
                rf_mm/=12
            print(rf_mm)
            return rf_mm
    return 0
    
    
# def func4(temp,hum,rain):
#     return model2.predict([[temp,hum,rain]])


lis=[]
def extract_parts(string):
    parts = string.split("___")
    lis.clear()
    lis.append(parts[0])
    print(parts)
    if(parts[1]=='healthy'):
        lis.append('Healthy')
    else:
        lis.append(parts[1])




@app.route('/hello', methods=['POST'])
def hello_world():
    data = request.json
    image_input_base64 = data["imageUrl"]
    predicted_class = predict_class_from_base64_image(image_input_base64, model, li)
    print(predicted_class)
    extract_parts(predicted_class)
    return jsonify(message=f"{predicted_class}",message1=f"{lis[0]}",message2=f"{lis[1]}")

@app.route('/hello2', methods=['POST'])
def hello_world2():
    data2 = request.json
    print(data2)
    if data2['question'] == "Why are my apple tree leaves turning brown and falling off?":
        response = "Leaves turning brown and falling off can be due to various factors like diseases or drought. Proper pruning and disease control can help."
    elif data2['question'] == "Why do my blueberry leaves look yellow and unhealthy?":
        response = "Yellow leaves may indicate nutrient deficiencies. Improve soil acidity and provide proper fertilization."
    elif data2['question'] == "How do I protect my cherry tree from diseases?":
        response = "Disease control and regular pruning are essential."
    elif data2['question']== "Why are the leaves on my corn plant turning brown at the tips?":
        response ="Brown tips on corn leaves might indicate underwatering or dry conditions. Ensure adequate moisture."
    elif data2['question']== "Why are the grape leaves discolored and spotted?":
        response = "Discolored and spotted leaves can result from various diseases. Use fungicides and maintain good air circulation."
    elif data2['question'] == "What is the best method for preventing pests on my orange tree?":
        response="Pesticides and proper care can prevent pests."
    elif data2['question'] =="How should I protect my peach tree from frost damage?":
        response = "Protect peach trees from frost with covers or frost cloth."
    elif data2['question'] =="Why do the pepper leaves have spots and discoloration?":
        response = "Spotted leaves can be caused by diseases; use fungicides and maintain good plant spacing for airflow."
    elif data2['question'] =="Why are the potato leaves wilting and turning yellow?":
        response = "Yellowing leaves can be due to overwatering or nutrient imbalances. Improve drainage and fertilization."
    elif data2['question'] =="How do I control weeds around my raspberry plants?":
        response = "Proper pruning and mulching can prevent weeds."
    elif data2['question'] =="Why are the soybean leaves turning yellow prematurely?":
        response = "Premature yellowing can indicate nutrient deficiencies; use balanced fertilization."
    elif data2['question'] =="What is the best way to control squash bugs and powdery mildew?":
        response = "Control squash bugs and powdery mildew with organic treatments."
    elif data2['question'] =="How should I winterize my strawberry plants?":
        response = "Winterize strawberries by mulching for protection."
    elif data2['question'] =="Why are my tomato leaves curling or showing signs of stress?":
        response = "Curling leaves can be due to various issues; ensure proper watering and balanced care."
    else:
        response = chatbot(data2['question'])


    return jsonify(message=f"{response}")

@app.route('/hello3', methods=['POST'])
def hello_world3():
    data3 = request.json
    plant_name = data3['DiseaseName']
    cure_message = get_plant_cure(plant_name)
    return jsonify(message=f"{cure_message}")

# @app.route('/hello4', methods=['GET'])
# def hello_world4():
#     print("in1")
#     temperature,humidity = func2()
#     return jsonify(temperature=temperature,humidity=humidity)

# @app.route('/hello5', methods=['POST'])
# def hello_world5():

#     data = {
#         "model": "gpt-3.5-turbo",
#         "messages": [
#             {"role": "user", "content": request.json['question']}
#         ]
#     }

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }

#     response = requests.post(api_url, data=json.dumps(data), headers=headers)
#     print(response.content)
#     response_data = json.loads(response.text)
#     content = response_data['choices'][0]['message']['content']
#     return jsonify(message=f"{content}")


@app.route('/hello6', methods=['POST'])
def hello_world6():
    data3 = request.json
    location = data3['location']
    print(location)
    rmff=func3(location)
    return jsonify(rainfall=rmff)

# @app.route('/hello7', methods=['POST'])
# def hello_world7():
#     data3 = request.json
#     print(data3)
#     temp=data3['temp']
#     hum=data3['hum']
#     rain=data3['rain']
#     num=func4(temp,hum,rain)
#     dictionary = {'Apple': 0, 'Banana': 1, 'Blackgram': 2, 'Chickpea': 3, 'Coconut': 4, 'Coffee': 5, 'Cotton': 6, 'Grapes': 7, 'Jute': 8, 'Kidneybeans': 9, 'Lentil': 10, 'Maize': 11, 'Mango': 12, 'Mothbeans': 13, 'Mungbean': 14, 'Muskmelon': 15, 'Orange': 16, 'Papaya': 17, 'Pigeonpeas': 18, 'Pomegranate': 19, 'Rice': 20, 'Watermelon': 21}
#     for key in dictionary.keys():
#         if dictionary[key] == num:
#             crop=key
#     return jsonify(crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
