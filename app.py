from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

# Load the model and preprocessor
preprocessor, model = joblib.load('LinearRegressionModel.pkl')

# Load cleaned car data
car = pd.read_csv('Cleaned_Car_data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = car.groupby('company')['name'].apply(list).to_dict()
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get form inputs
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Prepare data
        input_data = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Transform using preprocessor
        input_transformed = preprocessor.transform(input_data)

        # Predict
        prediction = model.predict(input_transformed)

        return f"₹{np.round(prediction[0], 2)}"

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)