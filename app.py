import streamlit as st
import pickle
from tensorflow.keras.models import load_model
import pandas as pd

st.title('Churn Prediction Model')

with open('scaler.pkl','rb') as File :
    scaler = pickle.load(File)

with open('gender.pkl','rb') as File:
    gender_ohe = pickle.load(File)

with open('ohe.pkl','rb') as File:
    ohe = pickle.load(File)

model = load_model('model.h5')



credit_score = st.number_input('Enter the credit score')

geography = st.selectbox('select the Geography',ohe.categories_[0])

gender = st.selectbox('Select the Gender',gender_ohe.categories_[0])

age = st.slider('Enter the Age',18,92)

tenure = st.slider('Select the tenure',0,10)

balance = st.number_input('Enter the balance')

num_of_product = st.slider('select the number of product',1,4)

has_cr_card = st.selectbox('select the has cr card',[0,1])

activate_member = st.selectbox('select the it is active member',[0,1])

estimated_salary = st.number_input('enter the number of estimated salary')




input_data = pd.DataFrame({

    'CreditScore':[credit_score],
    #'Geography':[geography],
    #'Gender':[gender_ohe.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_product],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[activate_member],
    'EstimatedSalary':[estimated_salary]
})


#input_data_df = pd.DataFrame([input_data])

ohe_geo_making = ohe.transform([[geography]]).toarray()
ohe_geo_df = pd.DataFrame(ohe_geo_making,columns=ohe.get_feature_names_out(['Geography']))

gender_making = gender_ohe.transform([[gender]]).toarray()
gender_making_df = pd.DataFrame(gender_making,columns=gender_ohe.get_feature_names_out(['Gender']))

# concatenate the data
final_df = pd.concat([input_data.reset_index(drop=True), ohe_geo_df, gender_making_df],axis=1)

# scaler

final_df_scaler = scaler.transform(final_df)

prediction = model.predict(final_df_scaler)
prob = prediction[0][0]

if st.button('prediction'):    
    st.write(f'churn probablitity {prob:2f}')

    if prob>0.5:
        st.write('it is more likely churn')

    else:
        st.write('it is not churn')

 

# prediction = model.predict(final_df_scaler)

# pred_prob = prediction[0][0]


# gender_ohe_making = gender_ohe.transform([[input_data['Gender']]])
# ohe_gender_df = pd.DataFrame(gender_ohe_making, columns=gender_ohe.get_feature_names_out())

#input_data_df = pd.DataFrame([input_data])

# df = pd.concat([input_data.drop(['Gender','Geography'],axis=1),ohe_geo_df,ohe_gender_df],axis=1)

# df_scaler = scaler.transform(df)


