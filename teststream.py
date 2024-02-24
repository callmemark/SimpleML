import streamlit as st
from core import MLMOddeler
from environment import *
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



######
## 
## PYTHON VERSION REQUIRED : 3.10 ^ 
##
######



# Initialized States
if "is_fitted" not in st.session_state:
	st.session_state.is_fitted = False


# cover full width
st.set_page_config(layout="wide")

# Main Title of the app
st.header("Simple :orange[ML]", divider="orange")

# Introduction Text
st.text("This is a simple implementation of Machine learning UI for classification problems powered by ScikitLearn and Pandas Library.")
st.text("Add a dataset below to use the application")





## Define Layouts Structures ##
row_1 = st.container(border=False)


row_2 = st.container(border=False)
r2_c1, r2_c2 = row_2.columns([3, 2])
r2_c1.subheader("Inspect Data", divider="orange")
r2_c2.subheader("Modify Data", divider="orange")


row_3 = st.container(border=False) 
r3_c1, r3_c2 = row_3.columns([1, 3])
r3_c1.subheader("Select Data", divider="orange")
r3_c2.subheader("Plot", divider="orange")


row_4 = st.container(border=False)
r4_c1, r4_c2 =  row_4.columns([2, 2])
r4_c1.subheader("Modify Pipeline", divider="orange")	
r4_c2.subheader("Testings & Reslts", divider="orange")


row_5 = st.container(border=False)
r5_c1, r5_c2 = row_5.columns([2,2])
r5_c1.subheader("Test Data", divider="orange")
r5_c2.subheader("Predict Playground", divider="orange")



## Define Constants ##
CHART_HEIGHT = 320 #px
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

INSPECT_DATA = {
	"Show All" : "_",
	"Summary of data" : "head"
}

MISSING_DATA_MODIFY = {
	"Fill NA/NaN values by using the next valid observation to fill the gap." : "bfill",
	"Remove missing values." : "dropna",
	"Fill NA/NaN values by propagating the last valid observation to next valid." : "ffil",
	"Fill NA/NaN values using the specified method." : "fillna",
	"Fill NaN values using an interpolation method" : "interpolate",

}


## ELEMENTS ##

# Create dataset input field 
input_dataset = row_1.file_uploader(label="Add dataset", type="csv")







## FUNCTIONS ##
def viewingData(df, callables_arr):
	df_copy = df.copy()

	for _callable in callables_arr:
		match INSPECT_DATA[_callable]:
			case "head":
				df_copy = df_copy.head()

			case "_":
				pass

			case _ :
				pass

	return df_copy


def processMissingData(df, callables_arr):
	df_copy = df.copy()

	for _callable in callables_arr:
		match MISSING_DATA_MODIFY[_callable]:
			case "bfill":
				df_copy = df_copy.bfill()

			case "dropna":
				df_copy = df_copy.dropna()

			case "ffil":
				df_copy = df_copy.ffil()

			case "fillna":
				df_copy = df_copy.fillna()

			case "interpolate":
				df_copy = df_copy.interpolate()

			case _ :
				pass

	return df_copy




def train(X_interest, y_interest):
	ml_model.splitTrainTest(X_interest, y_interest)
	st.session_state.is_fitted = True
	ml_model.fitData()

	calculated_accuracy = ml_model.performance()
	model_rating = "Accuracy of model: " + str(calculated_accuracy)
	r4_c2.write(model_rating)

	r5_c1.write(ml_model.X_test)





## RUN ON UPDATES ##
if input_dataset != None:
	df = pd.read_csv(input_dataset)
	header_columns = list(df.columns)

	r2_c2.text("Viewing data handling")
	data_selected_modification = r2_c2.multiselect("Select Functions", INSPECT_DATA.keys())
	r2_c2.caption("functions will be called in order of selection")

	r2_c2.text("Missing data handling")
	missing_data_modifications = r2_c2.multiselect("Select Functions", MISSING_DATA_MODIFY.keys())
	r2_c2.caption("functions will be called in order of selection")


	df = viewingData(df, data_selected_modification)
	df = processMissingData(df, missing_data_modifications)

	r2_c1.write(df)


	table_data_selected = r3_c1.multiselect("Select Columns", header_columns)
	table_type_select = r3_c1.selectbox("Table Type", CHART_TYPE)


	if(table_type_select == CHART_TYPE[0]):
		r3_c2.area_chart(data=table_data_selected, height=CHART_HEIGHT)

	elif (table_type_select == CHART_TYPE[1]):
		r3_c2.bar_chart(data=table_data_selected, height=CHART_HEIGHT)

	elif (table_type_select == CHART_TYPE[2]):
		r3_c2.line_chart(data=table_data_selected, height=CHART_HEIGHT)

	elif (table_type_select == CHART_TYPE[3]):
		r3_c2.scatter_cahrt(data=table_data_selected, height=CHART_HEIGHT)


	
	## Process Model
	ml_model = MLMOddeler(df)


	X_interest = r4_c2.multiselect("Select X Interests", header_columns)
	y_interest = r4_c2.selectbox("Select y Interests", header_columns)


	## creating custom pipeline
	scaler_selected = r4_c1.selectbox("Select Preprocessor", SCALERS.keys())
	algo_selected = r4_c1.selectbox("Select Algorithm", ALGORITHMS.keys())

	custom_pipeline = ml_model.CustomPipelLineModel(SCALERS[scaler_selected], ALGORITHMS[algo_selected])
	r4_c1.code(custom_pipeline, language="python", line_numbers=True)
	
	r4_c2.button(label="Train Model", on_click=train, kwargs={
			"X_interest" :  X_interest,
			"y_interest" : y_interest
			})

	ml_model.pipelLineModel(custom_pipeline)



	if st.session_state.is_fitted:
		if st.session_state.is_fitted:
			train(X_interest, y_interest)

		selected_row_test = r5_c2.number_input("Select row to test", value=None, placeholder="Select row from the Test Data table")

		if selected_row_test != None:
			try:
				test_row_data = pd.DataFrame(ml_model.X_test.iloc[int(selected_row_test)].values.reshape(1, -1), columns=ml_model.X_test.columns)

				r5_c2.write(test_row_data)
				r5_c2.text("Correct Answer: " + str(ml_model.y_test.iloc[int(selected_row_test)]))

				row_prediction = ml_model.model_pipeline.predict(test_row_data)[0]
				r5_c2.text("Predicted Answer: " + str(row_prediction))
			except Exception as error:
				r5_c2.code(error)
