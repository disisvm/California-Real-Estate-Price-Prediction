from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests

app = Flask(__name__)

# Load the model
with open('optimized_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the preprocessor
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    # Check for empty fields and set defaults
    garage_spaces = float(data['garageSpaces']) if data.get('garageSpaces') else 0

    features = {
        'livingArea^2': float(data['livingArea']) ** 2,
        'livingArea': float(data['livingArea']),
        'bathrooms': float(data['bathrooms']),
        'livingArea bedrooms': float(data['livingArea']) * float(data['bedrooms']),
        'bed_bath_rooms': float(data['bedrooms']) * float(data['bathrooms']),
        'living_area_per_room': float(data['livingArea']) / float(data['bedrooms']),
        'garageSpaces': garage_spaces,
        'bedrooms': float(data['bedrooms']),
        'bedrooms^2': float(data['bedrooms']) ** 2,
        'living_area_per_rooms': float(data['livingArea']) / float(data['bedrooms']),
        'hasGarage': int(data['hasGarage']),
        'longitude': float(data['longitude']),
        'living_area_per_bathroom': float(data['livingArea']) / float(data['bathrooms']),
        'yearBuilt': float(data['yearBuilt']),
        'latitude': float(data['latitude']),
        'garage_spaces_per_bedroom': garage_spaces / float(data['bedrooms']) if data.get('garageSpaces') else 0,
        'city': data['city'],
        'state': data['state'],
        'county': data['county'],
        'homeType': data['homeType']
    }

    input_df = pd.DataFrame([features])
    input_preprocessed = preprocessor.transform(input_df)
    prediction = model.predict(input_preprocessed)
    return jsonify({'prediction': prediction[0]})


@app.route('/geocode', methods=['GET'])
def geocode():
    address = request.args.get('address')
    api_key = '0b933d77854547548631b4ef2672904f'
    url = f'https://api.opencagedata.com/geocode/v1/json?q={address}&key={api_key}'
    response = requests.get(url)
    data = response.json()
    if data['results']:
        result = data['results'][0]
        return jsonify({
            'latitude': result['geometry']['lat'],
            'longitude': result['geometry']['lng'],
            'city': result['components'].get('city', ''),
            'state': 'CA',
            'county': result['components'].get('county', '')
        })
    return jsonify({'error': 'Address not found'})


if __name__ == '__main__':
    app.run(debug=True)
