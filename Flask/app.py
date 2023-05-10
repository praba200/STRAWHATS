from flask import Flask, render_template, request
import numpy as np
import pickle

app=Flask(__name__)#our flask app

@app.route('/')#rendering the html template
def home():
    return render_template('home.html')
@app.route('/predict')#rendering the html template
def index():
    return render_template("index.html")



@app.route('/data_predict',methods=['POST'])# route for our prediction
def predict():
    age=request.form['age']
    gender=request.form['gender']
    tb=request.form['tb']
    db=request.form['db']
    tp=request.form['tp']
    a1=request.form['a1']
    agr=request.form['agr']
    sgpt=request.form['sgpt']
    sgot=request.form['sgot']
    a2=request.form['a2']
    ip=request.form['ip']

#converting data into float format
data=[[float(age),float(gender),float(tb),float(db),float(tb),float(a1),float(agr),float(sgpt),float(sgot),float(a2),float(ip)]] 

#loading model which we saved
model=pickle.load(open("logregliverpatient.pkl",'rb'))

prediction=model.predict(data)[0]
if (prediction == 1):
    return render_template('nochance.html',prediction='You have a liver desease problem')
else:
    return render_template('chance.html',prediction='You dont have a liver desease problem')


if __name__=='__main__':
    app.run()
            

