from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from keras import Sequential
from keras.models import Sequential  
from sklearn.feature_extraction.text import CountVectorizer  
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import pickle
from model import *

app = Flask(__name__)

#Main
@app.route('/', methods=['GET', 'POST'])

def main():
    #loadedModel= pickle.load(open("class.model", "rb"))
    #loadedToken= pickle.load(open("tokenizer.pickle", "rb"))
    if request.method =="POST":
       # clf= joblib.load("SVM.pkl")
        clf= joblib.load("clf.pkl")
        tweet= request.form.get("tweet")
        input = [tweet]
        new_seq = tokenizer.texts_to_sequences(input)
        padded = pad_sequences(new_seq,
                         maxlen = 100,
                         padding = 'post',
                         truncating = 'post')
        pred="non"

        prediction= clf.predict(padded)
        if prediction ==[1]:
            prediction ="Rumor"
        else:
            prediction="Non-Rumor"

        #data=[tweet]
        #tokenizer = Tokenizer()
        #tok=tokenizer.fit_on_texts(data)
        #cv= CountVectorizer()
        #vect = cv.transform(data).toarray()



       # X= pd.DataFrame([[vect]], columns=["text"])

        #prediction= clf.predict(tok)
        #prediction= loadedModel.predict(loadedToken.fit_on_texts([tweet]))
    else:
        prediction=""
    return render_template("website.html", output= prediction, output2= tweet)

#Running the app

if __name__ == '__main__':
    app.run(debug= True)
