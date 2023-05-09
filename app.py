import pickle
import numpy as np
import pandas as pd
from flask import Flask, request,render_template,jsonify,url_for,app
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

regression = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('nav.html')


@app.route('/price_prediction',methods = ['POST'])
def price():
    data = request.json['data']
    inputdata = np.array(list(data.values())).reshape(1,-1)
    scaler  = StandardScaler()
    scaled = scaler.fit_transform(inputdata)
    pred = regression.predict(scaled)
    print(pred[0])
    return jsonify(pred[0])

@app.route('/prediction',methods = ['POST'])
def pred():
    data = [float(i) for i in request.form.values()]
    print(data)
    scaler = StandardScaler()
    final_input = scaler.fit_transform(np.array(data).reshape(1,-1))
    print(final_input)
    output  = regression.predict(final_input)[0]
    return render_template("index.html",result = "The House price is Estimated to be {}".format(output))

if __name__=="__main__":
    app.run(debug=True)