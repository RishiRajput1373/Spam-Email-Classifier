import streamlit as st
import polars as pl
import pickle

# Load model
with open("model/model.pkl",'rb') as file:
    model = pickle.load(file)

# Load dataset columns
data = pl.scan_csv('data/emails.csv')

feature_columns = data.drop(['Email No.', 'Prediction']).collect().columns

st.set_page_config(page_title="Spam Email Classifier")
st.title("📧 Email Spam Classifier")

st.write("Type your email text below")
# User email input
email_text = st.text_area(
    "Enter Email Message",
    height=200
)

# Predict Button

if st.button("Predict"):
    # Convert All Features to 0
    input_dict = dict.fromkeys(feature_columns, 0)
    #Split Words
    words = email_text.lower().split()

    #Count Words
    for word in words:
        if word in input_dict:
            input_dict[word] += 1

    #convert To DataFrame
    input_df = pl.DataFrame([input_dict]).to_pandas()

    #prediction 
    prediction = model.predict(input_df)[0]

    #result
    if prediction == 1:
        st.error("🚨 Spam Email")
    else:
        st.success("✅ Not Spam Email")