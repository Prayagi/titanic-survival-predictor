import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train.csv")
df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)

le_sex = LabelEncoder()
le_embarked = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

X = df.drop('Survived', axis=1)
y = df['Survived']

model = LogisticRegression(max_iter=200)
model.fit(X, y)

st.title(" Titanic Survival Predictor")
st.markdown("Enter passenger details to check if they would survive.")

pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses aboard", 0, 8, step=1)
parch = st.number_input("Number of Parents/Children aboard", 0, 6, step=1)
fare = st.number_input("Fare paid", 0.0, 600.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex_encoded = le_sex.transform([sex])[0]#conversion to encoded values
embarked_encoded = le_embarked.transform([embarked])[0]

#prdict
if st.button("Predict Survival"):
    input_data = [[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success(" :D This passenger would SURVIVE! 🎉")
        st.balloons()  
    else:
        st.error("❌ This passenger would NOT survive.")
