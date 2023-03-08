"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np

from sklearn.preprocessing import StandardScaler
from regression import (logreg, utils)



def test_prediction():
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
    )

    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

    # For testing purposes, once you've added your code.
    # CAUTION: hyperparameters have not been optimized.

	#train model
	log_model = logreg.LogisticRegressor(num_feats=6, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	log_model.train_model(X_train, y_train, X_val, y_val)

	#predict labels for test set
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
	y_pred = log_model.make_prediction(X_val)

	#check mean squared error is reasonable for some value
	mserror = np.mean((y_pred-y_val)**2)
	assert mserror < 0.5

	#check predictions are between 0 and 1
	assert np.min(y_pred) > 0
	assert np.max(y_pred) < 1

def test_loss_function():
	pass

def test_gradient():
	pass

def test_training():
	pass