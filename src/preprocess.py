import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    df = pd.read_csv(url, header=None, names=columns)

    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])
    df_clean = df.drop(columns=["species"])

    df.to_csv("C:/Users/6134155/Assignment-MLOPS/mlops-iris-pipeline/data/raw/iris.csv", index=False)
    df_clean.to_csv("C:/Users/6134155/Assignment-MLOPS/mlops-iris-pipeline/data/processed/iris_clean.csv", index=False)

    print("Data preprocessing complete.")

if __name__ == "__main__":
    load_and_preprocess()
