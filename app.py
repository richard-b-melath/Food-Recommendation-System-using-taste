from flask import Flask, render_template, request
import pickle
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)



# Load Universal Sentence Encoder 
model_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
model = hub.load(model_url)



print("Model loaded successfully")


# Load trained models and DataFrame from pickle file
with open('foody.pickle', 'rb') as f:
    data = pickle.load(f)
    nn = data['nn']
    pca = data['pca']
    df = data['df']

# Define a list of taste-related words for each taste category
taste_words = {
"ambient": ["ambient"],
    "astringent": ["astringent"],
    "astrigent": ["astrigent"],
    "barbecue": ["barbecue"],
    "briny": ["briny", "Briny"],
    "brittle": ["brittle"],
    "brothy": ["brothy", "Brothy"],
    "bitter": ["bitter"],
    "caramel": ["caramel"],
    "caramelized": ["caramelized", "Caramelized"],
    "charred": ["charred"],
    "chewy": ["chewy", "Chewy", "cummy", "elastic", "taffy-like"],
    "chilled": ["chilled", "chill"],
    "chunky": ["chunky", "Chunky"],
    "citrusy": ["citrusy", "Citrusy", "zesty"],
    "cool": ["cool"],
    "crispy": ["crispy", "Crispy", "crisp"],
    "crumbly": ["crumbly", "Crumbly"],
    "crunchy": ["crunchy", "Crunchy"],
    "creamy": ["creamy", "Creamy", "smooth", "Smooth"],
    "delicate": ["delicate", "Delicate"],
    "dense": ["dense", "Dense"],
    "dry": ["dry", "Dry"],
    "earthy": ["earthy", "Earthy", "nutty", "woody"],
    "elastic": ["elastic", "Elastic"],
    "flaky": ["flaky", "Flaky"],
    "floral": ["floral", "Floral"],
    "fluffy": ["fluffy", "Fluffy", "light and airy", "spongy"],
    "fragrant": ["fragrant", "Fragrant"],
    "fresh": ["fresh", "Fresh"],
    "fruity": ["fruity", "Fruity"],
    "full-bodied": ["full-bodied", "Full-bodied"],
    "gelatinous": ["gelatinous", "Gelatinous", "jelly-like", "soft and yielding", "wobbly"],
    "gummy": ["gummy", "Gummy"],
    "gooey": ["gooey", "Gooey"],
    "gritty": ["gritty", "Gritty"],
    "hard-frozen": ["hard-frozen", "Hard-frozen"],
    "hearty": ["hearty", "Hearty"],
    "herbal": ["herbal", "Herbal"],
    "herb-infused": ["herb-infused", "Herb-infused"],
    "hot": ["hot", "Hot", "steaming", "sizzling", "warm"],
    "icy": ["icy", "Icy", "ice"],
    "icy-cold": ["icy-cold", "Icy-cold"],
    "jelly": ["jelly", "Jelly"],
    "juicy": ["juicy", "Juciy"],
    "light": ["light", "Light"],
    "meaty": ["meaty", "Meaty"],
    "mildly warm": ["mildly warm", "Mildly warm"],
    "moist": ["moist", "Moist"],
    "neutral": ["neutral", "Neutral"],
    "nutty": ["nutty", "Nutty"],
    "refreshing": ["refreshing", "Refreshing"],
    "rich": ["rich", "Rich"],
    "roasted": ["roasted", "Roasted"],
    "room temperature": ["room temperature", "Room temperature"],
    "saccharine": ["saccharine", "Saccharine"],
    "salty": ["salty", "salt", "briny", "Salty", "savory"],
    "sizzling": ["sizzling", "Sizzling"],
    "sour": ["sour", "Sour", "tart"],
    "smoky": ["smoky", "Smoky"],
    "smooth": ["smooth", "Smooth"],
    "soft": ["soft", "Soft", "creamy", "tender"],
    "solid": ["solid", "Solid"],
    "spongy": ["spongy", "Spongy", "sponge-like"],
    "spicy": ["spicy", "Spicy", "spice"],
    "steamed": ["steamed", "Steamed"],
    "steam": ["steam", "Steam"],
    "sweety": ["sweety", "Sweet", "sweeting"],
    "sweet aroma": ["sweet", "Sweet", "sugary", "floral", "fruity"],
    "tangy": ["tangy", "Tangy"],
    "taffy": ["taffy", "Taffy"],
    "tart": ["tart", "Tart"],
    "textured": ["textured", "Textured"],
    "toasted": ["toasted", "Toasted"],
    "umami": ["umami", "Umami", "savory", "meaty", "full-bodied"],
    "varied": ["varied", "Varied"],
    "warm": ["warm", "Warm"],
    "woody": ["woody", "Woody"],
    "wobbly": ["wobbly", "Wobbly"],
    "zesty": ["zesty", "Zesty"]
}

def embed(texts):
    return model(texts)

def classify_taste(sentence):
    words = word_tokenize(sentence.lower())
    taste_counts = {taste: 0 for taste in taste_words}
    for word in words:
        for taste, taste_list in taste_words.items():
            if word in taste_list:
                taste_counts[taste] += 1
    predominant_tastes = [taste for taste, count in taste_counts.items() if count > 0]
    if predominant_tastes:
        return predominant_tastes
    else:
        return "No taste-related words found in the sentence."
def recommend(text):
    emb = embed([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]
    return df[["Food"]].iloc[neighbors].values.tolist()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    sentence = request.form['sentence']
    output = classify_taste(sentence)
    location = request.form['location']
    if output == "No taste-related words found in the sentence.":
        recommendation_found = False
        recommendation = None
    else:
        output = ', '.join(output)
        recm = recommend(output)
        recommendation_found = True
        recommendation = [{'food': food[0], 'combination': df.loc[df['Food'] == food[0], 'Combination'].values[0], 'loc': location} for food in recm]
    return render_template('index.html', recommendation_found = recommendation_found,recommendation=recommendation)    
if __name__ == '__main__':
    app.run(debug=True)
