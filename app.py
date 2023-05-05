import pickle
import numpy as np
import pandas as pd
from flask import Flask, request,render_template,jsonify,url_for,app
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

regression = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/price_prediction',methods = ['POST'])
def predict():
    data = request.json['data']
    inputdata = np.array(list(data.values())).reshape(1,-1)
    scaler  = StandardScaler()
    scaled = scaler.fit_transform(inputdata)
    pred = regression.predict(scaled)
    print(pred[0])
    return jsonify(pred[0])

if __name__=="__main__":
    app.run(debug=True)