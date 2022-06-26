import numpy as np
import pickle
import streamlit as st
import pandas as pd

loaded_model = pickle.load(open('regression_model.sav', 'rb'))


def welcome():
    return "Welcome All"


def predict_with_LG(input_data):
    new_array = np.asarray(input_data)
    input_data_reshaped = new_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'Iris-Setosa Species'
    elif prediction[0] == 1:
        return 'Iris-Versicolour Species'
    else:
        return 'Iris-Virginica Species'


def main():
    st.title('Iris Species Prediction System')
    
    #SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm
    SepalLengthCm = (st.text_input('Enter the length of the Sepal in cm:'))
    SepalWidthCm = (st.text_input('Enter the width of the Sepal in cm:'))
    PetalLengthCm = (st.text_input('Enter the length of the Petal in cm:'))
    PetalWidthCm = (st.text_input('Enter the width of the Petal in cm:'))

    predict = ''
    if st.button('Predict'):
        predict = predict_with_LG([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])
        b = np.array(predict, dtype=float) #  convert using numpy
        predict = [float(i) for i in predict]       
        
        #pd.to_numeric(predict)
    st.success(predict)


if __name__ == '__main__':
    main()
