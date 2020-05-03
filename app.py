from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

modelfile = open('car_dekho_model.pkl','rb')
model = pickle.load(modelfile)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])

def predict():
    print("Predict function is called")

    if request.method == 'POST':
        print(request.form.get('Year'))
        try:
            Year = float(request.form['Year'])
            PresentPrice = float(request.form['PresentPrice'])
            Kms  = float(request.form['Kms'])
            Owner  = float(request.form['Owner'])
            FuelType = float(request.form['FuelType'])
            SellerType  = float(request.form['SellerType'])
            Transmission   = float(request.form['Transmission'])

            pred_args = [Year, PresentPrice, Kms, Owner, FuelType, SellerType, Transmission]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)  

            model_pred = model.predict(pred_args_arr)
            model_pred = round(float(model_pred),3)  
        except:
            return "Please check if the values are filled correctly"

            pred_args = [Year, PresentPrice, Kms, Owner, FuelType, SellerType, Transmission]
            pred_args_arr = np.array(pred_args)

        '''pred_args = ['2010','3.6','2135','0','1','1','1']
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.astype(np.float)
        pred_args_arr = pred_args_arr.reshape(1,-1)  

        model_pred = model.predict(pred_args_arr)
        model_pred = round(float(model_pred),3)'''
    
    return render_template('predict.html', prediction = model_pred)



if __name__ == '__main__':
    app.run(host='0.0.0.0')
