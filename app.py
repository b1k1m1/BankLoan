# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:46:37 2020

@author: USHA
"""

import pandas as pd
from flask import Flask, request, render_template
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('model_new.pkl')
model._make_predict_function()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        
        if "firstname" in data:
            del data['firstname']
            
        if "lastname" in data:
            del data['lastname']
        
        if data['Gender'] == '1':
            data['Gender_Male'] = 1
            data['Gender_Female'] = 0
        else:
            data['Gender_Male'] = 0            
            data['Gender_Female'] = 1            
        if "Gender" in data:
            del data['Gender']
#          
        if data['Married'] == '1':
             data['Married_Yes'] = 1
             data['Married_No'] = 0
        else:
            data['Married_Yes'] = 0
            data['Married_No'] = 1        
        if "Married" in data:
            del data['Married']
 #           
            
        if data['Education'] == '1':
            data['Education_Graduate'] = 1
            data['Education_Not_Graduate'] = 0
        else:
            data['Education_Graduate'] = 0
            data['Education_Not_Graduate'] = 1            
        if 'Education' in data:
            del data["Education"]
            
        if data['Self_Employed'] == '1':
            data['Self_Employed_Yes'] = 1
            data['Self_Employed_No'] = 0
        else:
            data['Self_Employed_Yes'] = 0
            data['Self_Employed_No'] = 1        
        if 'Self_Employed' in data:
            del data['Self_Employed']
            
        if data['Property_Area'] == '1':
            data['Property_Area_Rural'] = 1
            data['Property_Area_Semiurban'] = 0            
            data['Property_Area_Urban'] = 0  
            
        if data['Property_Area'] == '2':
            data['Property_Area_Rural'] = 0
            data['Property_Area_Semiurban'] = 1
            data['Property_Area_Urban'] = 0 
            
        if data['Property_Area'] == '3':
            data['Property_Area_Rural'] = 0
            data['Property_Area_Semiurban'] = 0
            data['Property_Area_Urban'] = 1
            
        if 'Property_Area' in data:
            del data['Property_Area']
        
        newdf=pd.DataFrame(data, index =[0])
        df_np = newdf.to_numpy()
        unit=df_np.reshape(1,-1)
        sc_model = joblib.load('scalers.pkl')
        x_test = sc_model.transform(unit)
        y_predict1 = model.predict(x_test)
        y_predict1 = (y_predict1>0.50)
        newpred = pd.DataFrame(y_predict1, columns=['Status'])
        newpred = newpred.replace({True:'Approved', False:'Rejeted'})
        newpred1 = newpred.values[0][0]
                
       
        return render_template('index.html', prediction_text = "Your Loan has been {}".format(newpred1))

      
#        return '''<h2>The value of Dependents is: {}</h2>
#    			  <h2>The value of ApplicantIncome is: {}</h2>
#                  <h2>The value of CoapplicantIncome is: {}</h2> 
#                  <h2>The value of LoanAmount is: {}</h2>
#                  <h2>The value of Loan_Amount_Term is: {}</h2>
#                  <h2>The value of Credit_History is: {}</h2> 
#                  <h2>The value of Gender is: {}</h2>
#                  <h2>The value of Married is: {}</h2>
#                  <h2>The value of Education is: {}</h2>
#                  <h2>The value of Self_Employed is: {}</h2>
#                  <h2>The value of Property_Area_Rural is: {}</h2>
#                  <h2>The value of Property_Area_Semiurban is: {}</h2>
#                  <h2>The value of Property_Area_Urban is: {}</h2><br>'''.format(data['Dependents'], data['ApplicantIncome'], data['CoapplicantIncome'],
#																		   data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'], data['Gender'],
#																		   data['Married'], data['Education'], data['Self_Employed'], data['Property_Area_Rural'],
#                                                                           data['Property_Area_Semiurban'], data['Property_Area_Urban'])
#    
    
if __name__ == "__main__":
    app.run(debug=True)
    

