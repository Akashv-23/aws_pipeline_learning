import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,jsonify,render_template

application= Flask(__name__)
app=application

### import ridge regressor and stanadard scaler pickle
ridge_model=pickle.load(open("models/lassoCV.pkl","rb"))
scaler=pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def index():
  return render_template("index.html")          ## no need to name the folder (templates)  as render templates just take the files html        from templates automatically

@app.route("/predict",methods=["GET","POST"])
def prediction():
  if request.method=="POST":
    area=float(request.form.get("Area"))    ## the name in "get" should be matched from the home page input field "name"
    new_data=scaler.transform([[area]])
    result=ridge_model.predict(new_data)
    return render_template("home.html",results=result[0])     ## there are mutliple features 
  
  else:
    return render_template("home.html")         

if __name__=="__main__":
  app.run(host="0.0.0.0") 