from flask import Flask, request, render_template
import pickle
import numpy as np
import json
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the model, tokenizer, features, and captions
model = load_model(r'C:\Users\Kshitij navale\OneDrive\Desktop\image captioning\best_model.keras')
with open(r'C:\Users\Kshitij navale\OneDrive\Desktop\image captioning\tokenizer (1).pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open(r'C:\Users\Kshitij navale\OneDrive\Desktop\image captioning\features (1).pkl', 'rb') as f:
    features = pickle.load(f)

# Load captions from the JSON file
with open(r'C:\Users\Kshitij navale\OneDrive\Desktop\image captioning\captions.json', 'r') as f:
    captions = json.load(f)

max_length = 35  # Ensure this matches the max_length used during training

# Function to convert an index to a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Function to generate caption
def generate_caption(image_id):
    feature = features[image_id]
    feature = feature.reshape((1, -1))  # Reshape to (1, feature_length)
    return predict_caption(model, feature, tokenizer, max_length)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        image_id = file.filename.split('.')[0]
        
        # Save the file in the 'static/uploads/' folder
        img_path = os.path.join('static', 'uploads', file.filename)
        file.save(img_path)
        
        # Load the image
        image = Image.open(img_path)
        
        if image_id in features:
            caption = generate_caption(image_id)
            actual_captions = captions.get(image_id, ["No actual captions available."])
        else:
            caption = "No caption available for this image."
            actual_captions = ["No actual captions available."]
        
        # Send the image path relative to the 'static/' folder
        return render_template("result.html", image_path='uploads/' + file.filename, generated_caption=caption, actual_captions=actual_captions)

    return render_template("index.html")

if __name__ == "__main__":
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    app.run(debug=True)
