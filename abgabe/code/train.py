import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split


def import_df() -> pd.DataFrame:
    """
    imports dataset as pandas dataframe

    :return: pandas dataframe
    """
    df_imp = pd.read_csv('data\out\clean_mall_customer.csv')
    return df_imp


def metrics(data: pd.DataFrame, labels: np.ndarray) -> list:
    """
    calculates metrics silhouette, calinski and davies for clustering

    :param data: validation dataset
    :param labels: the clustering labels of the data in X
    :return: returns metrics for clustering
    """
    return [silhouette_score(data, labels),
            calinski_harabasz_score(data, labels),
            davies_bouldin_score(data, labels)]


def split_df(data: pd.DataFrame) -> list[pd.DataFrame]:
    """
    splits the dataframe into 3 dataframes at random, with distribution:
    train: 70%, validation: 20%, test: 10%
    
    :param data: X of model
    :return: X for train, test and validation
    """
    X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)

    # get 10% test data
    X_valid, X_test = train_test_split(X_test, test_size=1/3, random_state=42)

    return [X_train, X_test, X_valid]



def train(X: pd.DataFrame, X_val: pd.DataFrame, n_clusters: int = 10):
    """
    trains a clustering model using columns "ageincome" and "spending_scpre". 
    logs the metrics and params with mlfow for replicability.
    :param X: training dataset
    :param X_val: validation dataset
    :param n_clusters: number of clusters
    """
    
    with mlflow.start_run():
        model = KMeans(n_clusters=n_clusters, random_state=69)
        model.fit(X)
        predict = model.predict(X_val)

        silhouette, calinski, davies = metrics(X_val, predict)


        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("Silhouette", silhouette)
        mlflow.log_metric("Calinski-Harabasz", calinski)
        mlflow.log_metric("Davies-Bouldin", davies)

        mlflow.sklearn.log_model(model, "model")




def main():
    df = import_df()
    X_train, X_test, X_val = split_df(data = df[['ageincome','spending_score']])

    for i in range(20, 1, -1):
        train(X=X_train, X_val=X_val, n_clusters=i)



if __name__ == "__main__":
    main()
