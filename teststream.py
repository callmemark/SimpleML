import streamlit as st
from core import StellarModel
from environment import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# cover full width
st.set_page_config(layout="wide")

# Main Title of the app
st.header("Simple :orange[ML]", divider="orange")

# Introduction Text
st.text("This is a simple implementation of Machine learning UI for classification problems powered by ScikitLearn Library. Add a dataset below to use the application")






## Define Layouts Structures ##
row_1 = st.container(border=False)


row_2 = st.container(border=False)
r2_c1, r2_c2 = row_2.columns([2, 2])
r2_c1.subheader("Inspect Data", divider="orange")
r2_c2.subheader("Plot Data", divider="orange")


row_3 = st.container(border=False)
r3_c1, r3_c2 =  row_3.columns([2, 2])
r3_c1.subheader("Modify Model", divider="orange")	
r3_c2.subheader("Testings & Reslts", divider="orange")






## Define Constants ##
CHART_HEIGHT = 180 #px
CHART_TYPE = [
	"Area Chart",
	"Bar Chart",
	"Line Chart",
	"Scatter Chart",
	"Altair Chart"
]


SCALERS = {
	"Standard Scaler" : StandardScaler
}
	


ALGORITHMS = {
	"Random Forest Classifier" : RandomForestClassifier
}




## ELEMENTS ##

# Create dataset input field 
input_dataset = row_1.file_uploader(label="Add dataset", type="csv")



def train(X_interest, y_interest):
	stellar_model.splitTrainTest(X_interest, y_interest)
	stellar_model.fitData()

	calculated_accuracy = stellar_model.performance()
	model_rating = "Accuracy of model: " + str(calculated_accuracy)
	r3_c2.write(model_rating)



if input_dataset != None:
	df = pd.read_csv(input_dataset)
	r2_c1.write(df)
	header_columns = list(df.columns)

	table_data_selected = r2_c2.multiselect("Select Columns", header_columns)
	table_type_select = r2_c2.selectbox("Table Type", CHART_TYPE)

	if(table_type_select == CHART_TYPE[0]):
		r2_c2.area_chart(data=table_data_selected, height=CHART_HEIGHT)
	elif (table_type_select == CHART_TYPE[1]):
		r2_c2.bar_chart(data=table_data_selected, height=CHART_HEIGHT)
	elif (table_type_select == CHART_TYPE[2]):
		r2_c2.line_chart(data=table_data_selected, height=CHART_HEIGHT)
	elif (table_type_select == CHART_TYPE[3]):
		r2_c2.scatter_cahrt(data=table_data_selected, height=CHART_HEIGHT)



	X_interest = r3_c2.multiselect("Select X Interests", header_columns)
	y_interest = r3_c2.selectbox("Select y Interests", header_columns)


	## Process Model
	stellar_model = StellarModel(df)
	
	## creating custom pipeline
	scaler_selected = r3_c1.selectbox("Select Preprocessor", SCALERS.keys())
	algo_selected = r3_c1.selectbox("Select Algorithm", ALGORITHMS.keys())

	custom_pipeline = stellar_model.CustomPipelLineModel(SCALERS[scaler_selected], ALGORITHMS[algo_selected])
	
	stellar_model.pipelLineModel(custom_pipeline)
	



	

	r3_c2.button(label="Train Model", on_click=train, kwargs={
		"X_interest" :  X_interest,
		"y_interest" : y_interest
		})


