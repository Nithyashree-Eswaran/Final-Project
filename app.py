import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

# Load disease and supplement information
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Load the pre-trained model
model = CNN.CNN(39)  # 39 is the number of classes in the dataset
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()



def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

app = Flask(__name__)


@app.route('/get-alert-message')
def get_alert_message():
    return render_template('alert_msg.html')

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/about')
def about_page():
    return render_template('speech.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')





@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html', title=title, desc=description, prevent=prevent,
                               image_url=image_url, pred=pred, sname=supplement_name,
                               simage=supplement_image_url, buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']),
                           buy=list(supplement_info['buy link']))

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    transcript = data.get('transcript')
    if transcript:
        # Search disease_info and supplement_info based on the transcript
        search_results = []
        for idx, row in disease_info.iterrows():
            if transcript.lower() in row['disease_name'].lower():
                search_results.append({
                    'title': row['disease_name'],
                    'desc': row['description'],
                    'prevent': row['Possible Steps'],
                    'image_url': row['image_url']
                })

        for idx, row in supplement_info.iterrows():
            if transcript.lower() in row['supplement name'].lower():
                search_results.append({
                    'title': row['supplement name'],
                    'desc': '',  # Assuming no description for supplements
                    'image_url': row['supplement image'],
                    'buy_link': row['buy link']
                })

        return jsonify({'results': search_results})
    else:
        return jsonify({'status': 'error', 'message': 'No transcript received'}), 400

if __name__ == '__main__':
    app.run(debug=True)
