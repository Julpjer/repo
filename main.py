import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Fish Weight Prediction App")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("https://raw.githubusercontent.com/Julpjer/repo/master/Fish.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data

st.subheader("Please select relevant features of your fish!")
left_column, right_column = st.columns(2)
with left_column:
    inp_ps = st.radio(
        'Name of the fish:',
        np.unique(data['ps']))


input_Length1 = st.slider('Vertical length(cm)', 0.0, max(data["gewicht"]), 1.0)
input_Length2 = st.slider('Diagonal length(cm)', 0.0, max(data["voorrijs"]), 1.0)
input_Length3 = st.slider('Cross length(cm)', 0.0, max(data["rijen"]), 1.0)



if st.button('Make Prediction'):
    input_ps = encoder.transform(np.expand_dims(inp_ps, -1))
    inputs = np.expand_dims(
        [input_Length1, input_Length2, input_Length3, int(input_ps)], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your fish weight is: {np.squeeze(prediction, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")



