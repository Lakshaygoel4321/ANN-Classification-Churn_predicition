import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd


st.title('Chrun Prediction Model')

st.write('Solving the regression problem means finding is : estimated salary')

with open('ohe_reg.pkl','rb') as file:
    ohe_reg = pickle.load(file)

with open('le.pkl','rb') as file:
    label = pickle.load(file)

with open('scaler_reg.pkl','rb') as file:
    scaler = pickle.load(file)


model = load_model('model_reg.h5')


credit = st.number_input('Enter the Credit Score:')

gender = st.selectbox('select the gender',label.classes_)

geography = st.selectbox('Select the Geograpy area:',ohe_reg.categories_[0])

age = st.slider('Age',18,94)

tenure = st.slider('Tenure',0,10)

balance = st.number_input('Enter the balance')

num_of_product = st.slider('NumOfProduct',1,4)

has_cr_card = st.selectbox('select the hascrcredit:',[0,1])

is_active_member = st.selectbox('now it is active member of the organization',[0,1])

exited = st.selectbox('it is Exited',[0,1])


input_data = pd.DataFrame({

    'CreditScore':[credit],
    'Gender':[label.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'Exited':[exited]

})

# converting in the dataframe
#input_data_df = pd.DataFrame([input_data])

ohe_geo = ohe_reg.transform([[geography]]).toarray()
ohe_geo_df = pd.DataFrame(ohe_geo,columns=ohe_reg.get_feature_names_out(['Geography']))

final_df = pd.concat([input_data.reset_index(drop=True), ohe_geo_df],axis=1)

final_df_scaler = scaler.transform(final_df)

prediction = model.predict(final_df_scaler)
prob = prediction[0][0]

st.title(f'Prediction Estimated Salary: {prob:2f}')
