import streamlit as st
from src.pipeline.prediction_pipeline import PredictPipeline,Customdata
from src.logger import logging

logging.info('price prediction initiated')
st.set_page_config(page_title="Diamond Price",
                #    page_icon="",
                   layout='centered',
                   initial_sidebar_state='collapsed')
st.header("Diamond Price ")

col1, col2, col3= st.columns(3)  # Adjust the number of columns as needed

carat = col1.number_input("carat")
depth = col2.number_input("depth")
table = col3.number_input("table")

col4, col5, col6 = st.columns(3)
x = col4.number_input("x")

y = col5.number_input("y")

z = col6.number_input("z")

col7, col8, col9 = st.columns(3)

cut = col7.selectbox(
    "cut",
    ('Fair','Good','Very Good','Premium','Ideal'))
color = col8.selectbox(
    "color",
    ('D','E','F','G','H','I','J'))
clarity = col9.selectbox(
    'clarity',
    ("I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"))

submit = st.button('Submit')
logging.info("feature values extracted")
if submit:
    
    data = Customdata(carat=carat,
                      color=color,
                      clarity=clarity,
                      cut=cut,
                      table=table,
                      depth=depth,
                      z=z,
                      y=y,
                      x=x)
    data_as_df = data.get_data_as_dataframe()
    logging.info('predication initiated')
    predictor = PredictPipeline()
    prediction = predictor.predict(data_as_df)
    logging.info('prediction completed')
    st.write(prediction)