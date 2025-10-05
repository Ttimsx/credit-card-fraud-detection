import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
credited_card_df = pd.read_csv('credit_card.csv')
credited_card_df.head()
credited_card_df.info()
legal = credited_card_df[credited_card_df['Class']==0]
Illegal = credited_card_df[credited_card_df['Class']==1]
legal_sample = legal.sample(n = 492)
credited_card_df = pd.concat([legal_sample,Illegal], axis =0 )
# split data
# input fetaures =X
X = credited_card_df.drop('Class', axis =1)
# it removes the columnn name class from the table
# Y : independent features
Y = credited_card_df ['Class']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2, random_state = 20)
# # implementing logistic regression
# according to logistic regression, below 0.5 : toward 0 it will be legal
# else illegal
model = LogisticRegression()
model.fit(X_train,Y_train)
ypred  = model.predict(X_test) # predicted by logistic reg
accuracy_score(ypred,Y_test) # y_test is what we give to model to ensure  if it is able to predict
# accur
train_accuracy = accuracy_score(model.predict(X_train),Y_train)
test_accuracy = accuracy_score(model.predict(X_test),Y_test)
### web app
st.title("Credit Card Fraud Detection Model")
input_df = st.text_input("Enter all requires features value")
input_df_split = input_df.split(',')

submit = st.button("submit")
if submit:
    features = np.asanyarray(input_df_split, dtype = np.float64)
# we use reshape as we need ony one output either fraud or not fraud.
    prediction = model.predict(features.reshape(1,-1))
    if prediction[0]==0:
        st.write("legitimate Transaction")
    else:
        st.write("fraudulent transaction detected")