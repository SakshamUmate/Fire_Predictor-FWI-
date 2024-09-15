from flask import Flask, request,jsonify,render_template
import numpy as np
import pickle
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

application = Flask(__name__)
app=application
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predictor():
    if request.method=='GET':
        return render_template('Predict.html')
    elif request.method=='POST':
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ridge_model_path = os.path.join(current_dir, 'models', 'Algerian_ff_Ridge_model.pkl')
            scaler_path = os.path.join(current_dir, 'models', 'Algerian_ff_scaler.pkl')
            redge_model = pickle.load(open(ridge_model_path, 'rb'))
            Scaler = pickle.load(open(scaler_path, 'rb'))
            
            temp=float(request.form.get('Temperature'))
            rh=float(request.form.get('Rh'))
            ws=float(request.form.get('Ws'))
            rain=float(request.form.get('Rain'))
            ffmc=float(request.form.get('FFMC'))
            dmc=float(request.form.get('DMC'))
            isi=float(request.form.get('ISI'))
            classes=float(request.form.get('class'))
            region=float(request.form.get('Region'))
            
            
            
            scaled_data=Scaler.transform([[temp,rh,ws,rain,ffmc,dmc,isi,classes,region]])
            
            result=redge_model.predict(scaled_data)
            
            if result[0]>12:
                is_positive=True
            else:
                is_positive=False
            return render_template('Predict.html',Results=str(result[0]),is_positive=is_positive)
        except Exception as e:
            return f"Something went wrong \n \t {e}"
    else:
        return 'Invalid request '

if __name__ == '__main__':
    app.run(debug=True)
    