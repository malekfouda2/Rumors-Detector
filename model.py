import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import io
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import re
import string
from keras import models
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from keras.layers import LSTM, Dropout, Dense, Embedding
from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv1D, MaxPooling1D, Flatten , GRU   
from keras.models import Sequential    
from keras.optimizers import Adam
from sklearn.ensemble import VotingClassifier
import joblib
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import pickle

df= pd.read_csv("tweetinfo.csv", encoding = 'latin1')
labels=df.FinalLabel
df['clean_text']= df['text'].str.lower()


stop_words = stopwords.words('english')
df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df['clean_text'] = df['clean_text'].replace({';' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({',' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({'@' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({'$' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({'#' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({'&' : ''}, regex=True)
df['clean_text'] = df['clean_text'].replace({'\*' : ''}, regex=True)

df['clean_text'] = df['clean_text'].str.replace('http\S+|www.\S+', '', regex=True)




def tokenizeTweet(tweet):
     tok1= tokenizer.fit_on_texts(tweet)
#     # joblib.dump(tok1, "tokinizer.pkl")        
     
# tweet= df['clean_text']
# tokenizer= Tokenizer()
# word_index = tokenizer.word_index
# vocab_size = len(word_index)     
# tok1= tokenizer.fit_on_texts(tweet)
# sequences = tokenizer.texts_to_sequences(tweet)
# padded_seq = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

tweet= df['clean_text']
tokenizer= Tokenizer()

tokenizer.fit_on_texts(tweet)

word_index = tokenizer.word_index
vocab_size = len(word_index)
word_index
sequences = tokenizer.texts_to_sequences(tweet)
padded_seq = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')





embedding_index = {}
with open('glove.twitter.27B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

embedding_matrix = np.zeros((vocab_size+1, 50))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#X= df["clean_text"]
#Y= labels


x_train, x_test, y_train, y_test = train_test_split(padded_seq, df['FinalLabel'], test_size=0.1, random_state=42, stratify=df['FinalLabel'])


#SVM
parameters = { 
    'C': [1.0, 10],
    'gamma': [1, 'auto', 'scale']
}
modelSVM = GridSearchCV(SVC(kernel='rbf'), parameters, cv=5, n_jobs=-1)
modelSVM.fit(x_train, y_train)

#LGBM
params = {
    'learning_rate': 0.06,
    'n_estimators': 1500,
    'colsample_bytree': 0.5,
    'metric': 'f1_score'
}
modelLGBM = LGBMClassifier(**params)

#Fit or train the xgboost model
modelLGBM.fit(x_train.astype(np.float32), y_train, eval_set=[(x_train.astype(np.float32), y_train), (x_test.astype(np.float32), y_test)],
             verbose=400)

#Random Forest
modelRandomForest = RandomForestClassifier(n_estimators=200)

modelRandomForest.fit(x_train, y_train)


#LSTM
modelLSTM = Sequential([
    Embedding(vocab_size+1, 50, weights=[embedding_matrix], trainable=False),
    Dropout(0.2),
    LSTM(128),
    Dropout(0.2),
    Dense(100),
    Dense(1, activation='softmax')
])

batch_size = 5
epochs = 10
embeddingsize = 200

LR_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                             min_delta=0.001
                                            ,patience=10, 
                                            verbose=True,
                                            factor=0.1, 
                                            min_lr=0.001)
 

modelLSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

modelLSTM.fit(x_train,
              y_train, 
                    batch_size = batch_size , 
                    validation_data = (x_test, y_test) ,
                    epochs = epochs , 
                    callbacks = [LR_reduction])
                                            

#CNN
modelCNN = Sequential()
modelCNN.add(Embedding(vocab_size+1,50,weights=[embedding_matrix],trainable=False,input_length = 100))
modelCNN.add(Conv1D(filters=32,kernel_size=8,activation='relu'))
modelCNN.add(MaxPooling1D(pool_size=2))
modelCNN.add(Flatten())
modelCNN.add(Dense(10,activation='softmax'))
modelCNN.add(Dense(1,activation='sigmoid'))
    
modelCNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
modelCNN.fit(x_train,y_train,epochs=20)


#GRU
emb_dim = embedding_matrix.shape[1]
gru_model = Sequential()
gru_model.add(Embedding(vocab_size+1, 50 , weights=[embedding_matrix], trainable = False))
gru_model.add(GRU(128, return_sequences=False))
gru_model.add(Dropout(0.5))
gru_model.add(Dense(1, activation = 'sigmoid'))
gru_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 256
epochs  = 10
historyGRU = gru_model.fit(x_train,y_train,  batch_size = batch_size, epochs = epochs)


#LSTM WITH CNN
embedding_vector_length = 32
cnn_model = Sequential()

cnn_model.add(Embedding(vocab_size+1, 50 , weights=[embedding_matrix], trainable = False))
cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(LSTM(100))
cnn_model.add(Dense(units=1, activation='sigmoid'))

cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_c=cnn_model.fit(x_train,y_train, epochs=5, batch_size=64,verbose = 1,)





#Voting Classifier
#clf= VotingClassifier(estimators=[ ('LGBM', modelLGBM), ('Random Forest', modelRandomForest),
           #                       ('SVM', modelSVM)  ],
              #          voting='hard')
#clf.fit(x_train, y_train)


#joblib.dump(clf, "clf.pkl")

joblib.dump(modelSVM, "SVM.pkl")
#pickle.dump(modelSVM, open("class.model", "wb") )
#pickle.dump(tok1, open("tokenizer.pickle", "wb"))
#joblib.dump(tok1, "tokinizer.pkl")
joblib.dump(modelLSTM,"lstm.pkl")
