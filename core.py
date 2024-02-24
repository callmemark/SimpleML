import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np


class MLMOddeler():
	def __init__(self, df):
		self.df = df
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None
		self.model_pipeline = None
		self.clf = None


	def splitTrainTest(self, train_col, label):
		# seperate training data to test data
		X = self.df[train_col]
		y = self.df[label]

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0)
		print("Data split complete")


	def pipelLineModel(self, custom_pipeline):
		# pipeline processes
		if custom_pipeline != None:
			self.model_pipeline = custom_pipeline
		else:
			self.model_pipeline = make_pipeline(
				StandardScaler(),
				RandomForestClassifier(n_estimators=10)
			)

		print("pipe line created")


	def CustomPipelLineModel(self, scaler_callable, classifier_callable):
		# pipeline processes
		custom_model_pipeline = make_pipeline(
			scaler_callable(),
			classifier_callable()
		)

		return custom_model_pipeline



	def fitData(self):
		# fit the training data
		self.clf = self.model_pipeline.fit(self.X_train, self.y_train)



	def performance(self):
		# create prediction
		prediction = self.clf.predict(self.X_test)

		# meassure accuracy
		accuracy = np.mean(prediction == self.y_test)
		return accuracy