import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
 
# Load preprocessed data
df = pd.read_csv("C:/Users/Harish Kumar/Downloads/mlops-iris-pipeline/data/processed/iris_clean.csv")
X = df.drop(columns=["species_encoded"])
y = df["species_encoded"]
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Set MLflow experiment
mlflow.set_experiment("Iris_Classification")
 
def train_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
 
        # Log parameters and metrics
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)
 
        # Log model with input example
        input_example = X_test.iloc[:1]
        mlflow.sklearn.log_model(model, artifact_path=model_name, input_example=input_example)
 
        print(f"{model_name} Accuracy: {acc:.4f}")
 
# Train models
train_model(LogisticRegression(), "LogisticRegression")
train_model(RandomForestClassifier(), "RandomForest")