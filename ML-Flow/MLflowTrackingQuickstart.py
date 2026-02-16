# MLflow Tracking Quickstart
# https://mlflow.org/docs/latest/ml/getting-started/quickstart/

import mlflow

mlflow.set_experiment("MLflow Quickstart-Second")


import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,  roc_auc_score 

#Step 2 - Prepare training data
#Before training our first model, let's prepare the training data and model hyperparameters.

# Load the Iris datasetmlflow server --port 5000

X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 8888,
}


#Step 3 - Train a model with MLflow Autologging
#In this step, we train the model on the training data loaded in the previous step, and log the model and its metadata to MLflow. The easiest way to do this is to using MLflow's Autologging feature.

import mlflow

# Enable autologging for scikit-learn
mlflow.sklearn.autolog()


# Just train the model normally
lr = LogisticRegression(**params , class_weight='balanced' , fit_intercept=True)
lr.fit(X_train, y_train)

"""
Step 5 - Log a model and metadata manually
Now that we have learned how to log a model training run with MLflow autologging, let's step further and learn how to log a model and metadata manually. This is useful when you want to have more control over the logging process.

The steps that we will take are:

Initiate an MLflow run context to start a new run that we will log the model and metadata to.
Train and test the model.
Log model parameters and performance metrics.
Tag the run for easy retrieval.
"""
# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Train the model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Log the model
    model_info = mlflow.sklearn.log_model(sk_model=lr, name="iris_model")

    # Predict on the test set, compute and log the loss metric
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # Optional: Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info - Rayn", "Basic LR model for iris data - Ryan")

    """
    Step 6 - Load the model back for inference.
    After logging the model, we can perform inference by:

    Loading the model using MLflow's pyfunc flavor.
    Running Predict on new data using the loaded model.
    """
    # Load the model back for predictions as a generic Python Function model
    loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

    predictions = loaded_model.predict(X_test)

    iris_feature_names = datasets.load_iris().feature_names

    result = pd.DataFrame(X_test, columns=iris_feature_names)
    result["actual_class"] = y_test
    result["predicted_class"] = predictions

    print(result[:4])