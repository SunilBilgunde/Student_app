import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_model():
    with open("student_lr_final_model.pkl",'rb') as file:
        model,scaler, Le = pickle.load(file)
    return model,scaler,Le

def preprocessing_input_data(data,scaler,Le):
    data["Extracurricular Activities"] = Le.transform([data["Extracurricular Activities"]])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,Le = load_model()
    processed_data = preprocessing_input_data(data,scaler,Le)
    prediction =  model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance prediction")
    st.write("Enter your data to get a prediction of your performance")

    Hours_Studied = st.number_input("Hours studied",min_value=1,max_value=10,value=6)
    Previous_Score = st.number_input("Previous score",min_value=35,max_value=100,value=70)
    Extra_Curr_Activities = st.selectbox("Extra Curricular activity",['Yes', 'No'])
    Sleeping_duration = st.number_input("Sleeping Hours",min_value=5,max_value=10,value=7)
    Question_paper_solved = st.number_input("Number of question papers solved",min_value=0,max_value=10,value=4)

    if st.button("Predict"):
        user_data = {
            "Hours Studied":Hours_Studied,
            "Previous Scores":Previous_Score,
            "Extracurricular Activities":Extra_Curr_Activities,
            "Sleep Hours":Sleeping_duration,
            "Sample Question Papers Practiced":Question_paper_solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your prediction result is {prediction}")


if __name__ == "__main__":
    main()