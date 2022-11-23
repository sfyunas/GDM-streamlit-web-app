'''
    Front-end for GDM web-app using streamlit library
    
    Author: Syed Fahad Yunas
    
    Date: 22 November 2022
'''

#---------------------
# Import dependencies
#---------------------

import streamlit as st
import pandas as pd
import joblib


#----------------------------
# Loading the model pipeline
#----------------------------

model = joblib.load('model_rf.joblib')


#------------------------------------
# Creating a web-page for our web-app
#------------------------------------


# Adding title and subtitle
st.title("Gestational Diabetes Mellitus (GDM) prediction")
st.subheader("Enter expectant mother's information and submit for predicting GDM likelihood")

# Creating a form to collect inputs
with st.form(key='my_form'):
    age = st.number_input('Age', min_value=10, max_value=100, value=10, step=1)
    no_of_pregnancy = st.number_input('Number of pregnancy', min_value=0, max_value=20, value=0, step=1)
    gestation_in_previous_pregnancy = st.number_input('Gestation in previous birth? (0 for No, 1 for Yes)', min_value=0, max_value=1, value=0, step=1)
    bmi = st.number_input('Body Mass Index?', min_value=9, max_value=65, value=20, step=1)
    hdl = st.number_input('High Density Lipo-protein', min_value=10, max_value=100, value=50, step=1)
    family_history = st.number_input('Family history of Diabetes? (0 for No, 1 for Yes)', min_value=0, max_value=1, value=0, step=1)
    unexplained_prenetal_loss = st.number_input('Unexplained prenatal loss in previous pregnancies?(0 for No, 1 for Yes)', min_value=0, max_value=1, value=0, step=1)
    large_child_or_birth_default = st.number_input('Large Child (current pregnancy)(0 for No, 1 for Yes)?', min_value=0, max_value=1, value=0, step=1)
    pcos = st.number_input('Have poly-cystic ovarian syndrome(0 for No, 1 for Yes)?', min_value=0, max_value=1, value=0, step=1)
    sys_bp = st.number_input('Systolic BP', min_value=100, max_value=200, value=100, step=1)
    dia_bp = st.number_input('Diastolic BP', min_value=60, max_value=120, value=65, step=1)
    ogtt = st.number_input('Oral Glucose Tolerance Test', min_value=50, max_value=400, value=200, step=1)
    hemoglobin = st.number_input('Hemoglobin (HB) value', min_value=10, max_value=15, value=11, step=1)
    sedentary_lifestyle = st.number_input('Do you have sedentary lifestyle?(0 for No, 1 for Yes)', min_value=0, max_value=1, value=0, step=1)
    prediabetes = st.number_input('Did you have diabetes before pregnancy?(0 for No, 1 for Yes)', min_value=0, max_value=1, value=0, step=1)
    submit_button = st.form_submit_button(label='Submit for prediction')
    
if submit_button:
    # storing our input data in a dataframe
    input_df = pd.DataFrame({
        "age": [age],
        "no_of_pregnancy": [no_of_pregnancy],
        "gestation_in_previous_pregnancy": [gestation_in_previous_pregnancy],
        "bmi": [bmi],
        "hdl": [hdl],
        "family_history": [family_history],
        "unexplained_prenetal_loss": [unexplained_prenetal_loss],
        "large_child_or_birth_default": [large_child_or_birth_default],
        "pcos": [pcos],
        "sys_bp": [sys_bp],
        "dia_bp": [dia_bp],
        "ogtt": [ogtt],
        "hemoglobin": [hemoglobin],
        "sedentary_lifestyle": [sedentary_lifestyle],
        "prediabetes": [prediabetes]
    })

    # Apply the model pipeline to our input data and extract prediction
    pred_proba = model.predict_proba(input_df)[0][1]

    # Output prediction
    st.subheader(f"Based on the information provided, there is {pred_proba:.0%} likelihood that the expecting mother might develop gestational diabetes.")
    if pred_proba >= 0.4:
        st.write("It is advisable to refer the expecting mother to a GP for further diagnosis.")

