import numpy as np
import pickle
import streamlit as st
import pandas as pd

#Loading the RandomForest model in format of pkl file
loaded_model = pickle.load(open('rfclassifier_model.sav', 'rb')

def welcome():
  return 'Welcome All'
                           
                           

def predict_with_rf(input_data):
  new_array = np.asarray(input_data).reshape(1, -1)
                           
  prediction = loaded_model.predict(new_array)
  if prediction[0] == 0:
                           return 'Iris-Setosa Species'
  elif prediction[0] == 1:
                           return 'Iris-Versicolor Species'
  else:
                           return 'Iris-Virginica Species'
                           
                           
                           
                           
def main():
  st .title("Iris Species prediction using Random Forest Algorithm")
  
  predict = ''
  
  SepalLengthCm = st.text_input('Enter the length of the sepal in Cm:')
  SepalWidthCm = st.text_input('Enter the width of the sepal in Cm:')
  PetalLengthCm = st.text_input('Enter the length of the petal in Cm:')
  PetalWidthCm = st.text_input('Enter the width of the petal in Cm:')
  
  if st.button('Predict'):
                           predict = predict_with_rf([SpealLengthCm, SpealWidthCm, PetalLengthCm, PetalWidthCm])
  st.success(predict)

                           

if __name__ == '__main__':
  main()
