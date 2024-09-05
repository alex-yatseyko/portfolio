import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('../model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']
country_encoder = data['country_encoder']
education_encoder = data['education_encoder']

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""#### We need some information to predict the salary""")

    countries = (
        "Other",
        "United States of America",
        "Germany",
        "United Kingdom of Great Britain and Northern Ireland",
        "Ukraine",
        "India",
        "France",
        "Canada",
        "Brazil",
        "Spain",
        "Italy",
        "Netherlands",
        "Australia"
    )

    educations = (
        'Post grad', 'Master degree', 'Less than a Bachelor', 'Bachelor degree'
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", educations)

    experience = st.slider("Years of experience", 0, 50, 3)

    ok = st.button('Calculate Salary')
    if ok:
        X = np.array([[country, education, experience]])
        X[:,0] = country_encoder.transform(X[:,0])
        X[:,1] = education_encoder.transform(X[:,1]) 
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${int(salary[0])}")
