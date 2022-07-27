import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_forest(Year,Mileage):
    input=np.array([[Year,Mileage]]).astype(np.float64)
    prediction=model.predict(input)
    # pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(prediction)

def main():
    st.title("Streamlit Tutorial")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Car Pricr Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    Year = st.text_input("Year","Type Here")
    Mileage = st.text_input("Mileage","Type Here")



    if st.button("Predict"):
        output=predict_forest(Year,Mileage)
        st.success(output)


if __name__=='__main__':
    main()