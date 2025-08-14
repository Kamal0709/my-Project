import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import sqlite3

import numpy as np
import pandas as pd
from sklearn import metrics 
import warnings
import pickle
import pandas as pd
import numpy as np
import pickle
import sqlite3
import random

import smtplib 
from email.message import EmailMessage
from datetime import datetime

warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Dropout, Flatten, LSTM, Conv1D
model = tf.keras.models.load_model(
    'model.h5',
    compile=False,
    custom_objects={
        'GRU': GRU,
        'Bidirectional': Bidirectional,
        'Dense': Dense,
        'Dropout': Dropout,
        'Flatten': Flatten,
        'LSTM': LSTM,
        'Conv1D': Conv1D
    }
)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route('/home1')
def home1():
	return render_template('home1.html')


@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")

@app.route("/notebook")
def notebook1():
    return render_template("GasDetection.html")



@app.route('/predict',methods=['POST'])
def predict():
    int_features= [float(x) for x in request.form.values()]
    print(int_features,len(int_features))
    dj = np.asarray(int_features)
    dj = dj.reshape(-1,11,1)
    prediction_proba = model.predict(dj)
    predict=np.argmax(prediction_proba,axis=1)

    if predict==0:
        output='Methane Based Gas Detected!'
    elif predict==1:
        output='Sulphur Based Gas Detected !'
    
    

    return render_template('prediction.html', output=output)



if __name__ == "__main__":
    app.run(debug=True)

