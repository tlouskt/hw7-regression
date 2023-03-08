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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
	""" Check against sklearn's logistic regression loss """
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
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

	test_loss = log_model.loss_function(y_val, log_model.make_prediction(X_val))
	sklearn_loss = log_loss(y_val, log_model.make_prediction(X_val))

	assert np.allclose(test_loss, sklearn_loss) == True

def test_gradient():
	""" Check gradient values against what is calculated """
	# Create log reg, test set and set weights
	logr = logreg.LogisticRegressor(num_feats=4, learning_rate=0.00001, tol=0.01, max_iter=10, batch_size=10)
	X = np.array([[1,2,3,4], [5,6,7,8,], [1, 0.5, 0.3, 0.2]])
	y = np.array([1,0, 1])
	logr.W = np.array([1, 1, 1])

	#calculate gradient
	calc_grad = logr.calculate_gradient(X,y)
	check_grad = np.array([-0.07946861, -0.24613528, -0.51280195, -0.81280195])
	assert np.allclose(calc_grad, check_grad)


def test_training():
	""" Check weights are being updated """
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
	X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

	loss_vals = log_model.loss_hist_train
	avg_init_loss = np.mean(loss_vals[:20])
	avg_final_loss = np.mean(loss_vals[20:])
	#check loss decreasing, so check average first 20 loss values and las 20 value. 
	# Unit test fails if just checking individual. expect loss value averages to drop over time and be smaller. 
	# This also checks the gradient is working as expected

	assert avg_init_loss >= avg_final_loss


