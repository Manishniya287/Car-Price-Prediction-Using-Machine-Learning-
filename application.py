csvfrom flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import json

app = Flask(__name__)

car = pd.read_csv("Cleaned_Car_data.csv")

# Load trained model and data
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    car_models = car.groupby('company')['name'].unique().apply(list).to_dict()
    companies.insert(0, "Select Company")
    
    return render_template(
        'index.html',
        companies=companies,
        car_models=json.dumps(car_models),  # Pass car_models as JSON
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    print(company, car_model, year, fuel_type, kms_driven)  # Debugging log

    # Create DataFrame with the input features
    input_df = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Prediction
    prediction = model.predict(input_df)

    return str(np.round(prediction[0], 2))  # Fix: Access first element of prediction

if __name__ == "__main__":
    app.run(debug=True)
