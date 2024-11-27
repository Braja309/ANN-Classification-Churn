import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

# Load the trained Model
model = tf.keras.models.load_model('model.h5')

# Load Encoder and Scaler
with open('LabelEncoder.pkl','rb') as file:
    Label_Encoder_Gender = pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    Onehot_Encoder_Geo = pickle.load(file)

with open('scaler_churn.pkl','rb') as file:
    Scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Input Data
creditScore = st.number_input('Credit Score')
gender=st.selectbox('Gender',Label_Encoder_Gender.classes_)
age=st.slider('Age',18,90)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('balance')
numOfProducts=st.slider('Number of Product',1,4)
hasCrCard=st.selectbox('Has Credit card',[0,1])
isActiveMember=st.selectbox('Is Active Member',[0,1])
estimatedSalary=st.number_input('Estimated Salary')
geography=st.selectbox('Geography',Onehot_Encoder_Geo.categories_[0])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [creditScore],
    'Gender': [Label_Encoder_Gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [numOfProducts],
    'HasCrCard': [hasCrCard],
    'IsActiveMember': [isActiveMember],
    'EstimatedSalary': [estimatedSalary]
})

# One-hot encode 'Geography'
geo_encoded = Onehot_Encoder_Geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=Onehot_Encoder_Geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = Scaler.transform(input_data)


# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
