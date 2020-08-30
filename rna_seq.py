from collections import defaultdict

from scipy.stats import stats
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import math
import seaborn as seabornInstance
import seaborn as sns  # Plotting library
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RANSACRegressor, Ridge
from sklearn import metrics, linear_model, neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn.metrics import mean_squared_error


#def f(x):
 #   return np.int(x)


#def rmse(predictions, targets):  # root mean squared error
 #   predictions = predictions.astype(int)
  #  targets = targets.astype(int)
   # f3 = np.vectorize(f)
    #return (np.sqrt(np.mean((float((predictions)) - ((targets))) ** 2)))


def agglomerativeClustering():

    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.cluster.hierarchy as shc
    from sklearn.cluster import AgglomerativeClustering

    dataset = read_df()
    dataset = dataset[['BindLevel', 'Gl', 'Gp', 'Ip']].values
    data = dataset

    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    plt.show()

    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)

    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()



def read_df():
    # Convert to DB.. style names
    df = pd.read_csv("./SRR057629.txt", sep="\t")
    df = df.replace(to_replace="WB", value="0")
    df = df.replace(to_replace="SB", value="1")
    df = df.replace(to_replace="Q20;badReads", value="0")
    df = df.replace(to_replace="Q20", value="0")
    df = df.replace(to_replace="badReads", value="0")
    df = df.replace(to_replace="PASS", value="1")
    df = df.replace(to_replace="SC;alleleBias", value="0")
    df = df.replace(to_replace="SC", value="0")
    df = df.replace(to_replace="alleleBias", value="0")

    pd.set_option('display.max_columns', None)
    df = df.fillna(0)
    print(df.tail())


    # df = pd.read_csv("./matching.txt", sep="\t")

    #test_data_length_9 = pd.read_csv("./Prostate_test_data_length_9.txt", sep="\t")
    # df = test_data_length_9 # for test data

    return df


def k_neihgbours(dataset):

    # Source:https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

    # Preprocessing, split dataset into attributes and labels
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, (len(list(dataset))-1)].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Feature Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Training and Predictions
    classifier = KNeighborsClassifier(n_neighbors=40)
    classifier.fit(X_train, y_train)

    # Make predictions on our test data
    y_pred = classifier.predict(X_test)

    # Print separating line
    print("." * 80)
    print("Predictions -----")

    # Evaluating the Algorithm
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Comparing Error Rate with the K Value
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()


def regression(regressor, x_name, y_name,df):

    # Make plot of distribution
    df.plot(x=x_name, y=y_name, style='o')
    plt.title(x_name + " vs " + y_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.tight_layout()
    seabornInstance.distplot(df[y_name])

    # Define train and test data
    X = df[x_name].values.reshape(-1, 1)
    y = df[y_name].values.reshape(-1, 1)

    # Split data in sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor.fit(X_train, y_train)  # training the algorithm

    # Print separating line
    print("." * 80)
    print("Predictions -----" + " " + regressor.__class__.__name__)

    # For retrieving score
    # Returns the coefficient of determination R^2 of the prediction
    print("Score: ", regressor.score(X, y, sample_weight=None))
    print("RMSE: ", mean_squared_error(X_train, y_train))


def main():
    df = read_df()
    #df = pd.DataFrame(df)

    x_name = "BindLevel"
    y_name = "Rank"

    X = df[x_name].values
    y = df[y_name].values

    regression(LinearRegression(), x_name, y_name, df)
    regression(Ridge(alpha=.5), x_name, y_name, df)
    regression(neighbors.KNeighborsRegressor(), x_name, y_name, df)
    regression(DecisionTreeRegressor(random_state=0), x_name, y_name, df)
    #regression(RANSACRegressor(random_state=0), x_name, y_name, df)
    regression(VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=10, random_state=1))]), x_name, y_name, df)



    # Selecting columns
    dataset = df[['BindLevel', 'Gl', 'Gp', 'Ip', 'Mixcr']]

    k_neihgbours(dataset)

    #agglomerativeClustering()



if __name__ == '__main__':
    import sys
    import warnings

    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    main()


