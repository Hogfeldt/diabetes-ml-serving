from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib 
import numpy as np

from transform_into_numeric import read_encoding_file

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

model = joblib.load("data/model.joblib")
encoding = read_encoding_file('data/data_types.txt')
features = [
    'race', 
    'gender', 
    'age', 
    'time_in_hospital', 
    'num_lab_procedures',
    'num_procedures',
    'num_medications',
    'number_outpatient', 
    'number_emergency', 
    'number_inpatient', 
    'number_diagnoses', 
    'max_glu_serum', 
    'A1Cresult', 
    'metformin', 
    'repaglinide', 
    'nateglinide', 
    'chlorpropamide', 
    'glimepiride', 
    'acetohexamide', 
    'glipizide', 
    'glyburide', 
    'tolbutamide', 
    'pioglitazone', 
    'rosiglitazone', 
    'acarbose', 
    'miglitol', 
    'troglitazone', 
    'tolazamide', 
    'examide', 
    'citoglipton',
    'insulin', 
    'glyburide_metformin', 
    'glipizide_metformin', 
    'glimepiride_pioglitazone', 
    'metformin_rosiglitazone', 
    'metformin_pioglitazone', 
    'change', 
    'diabetesMed', 
    'readmitted', 
    '_diag_1', 
    '_diag_2', 
    '_diag_3'
    ]

def json_to_num_array(data):
    array = list()
    print(data)
    for feature in features:
        print(feature)
        array.append(str(data[feature]))
    out = [fn(value) for value, fn in zip(array, encoding)]
    out = out[:3] + out[4:] # remove label
    return out

def eval_model(data):
    X =  np.array([data])
    print(X)
    return {"prediction": model.predict(X)[0]}

@app.route('/')
def hello_world():
    return('Hello, World!')

@app.route('/api/predict', methods=["POST"])
def predict():
    data = request.get_json()
    data_array = json_to_num_array(data['patient'])
    resp = eval_model(data_array) 
    return(jsonify(resp))
