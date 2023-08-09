# Libraries
import warnings


## Data processing
import pandas as pd, numpy as np

## Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

## Model
from sklearn.linear_model import LogisticRegression


# Preprocessing
def readFile(path):
    data = pd.read_csv(path)
    return data


def xySplitter(data):
    cols = data.columns
    X = data[cols[:-1]]
    y = data[cols[-1]]
    return X, y

# Training
def trainTest(x, y):
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )
    return X_train, X_test, y_train, y_test


def modeling(model, X_train, y_train):
    return model.fit(X_train, y_train)

# Predicting
def predictData(model, data):
    prediction = model.predict(data)
    # accuracy = accuracy_score(y_test, prediction)
    
    print("=" * 60)
    if prediction == 13:
        print('Crop Recomendation: rice')
    elif prediction == 9:
        print('Crop Recomendation: maize')
    elif prediction == 0:
        print('Crop Recomendation: soybeans')
    elif prediction == 3:
        print('Crop Recomendation: beans')
    elif prediction == 12:
        print('Crop Recomendation: peas')
    elif prediction == 8:
        print('Crop Recomendation: groundnuts')
    elif prediction == 6:
        print('Crop Recomendation: cowpeas')
    elif prediction == 2:
        print('Crop Recomendation: banana')
    elif prediction == 10:
        print('Crop Recomendation: mango')
    elif prediction == 7:
        print('Crop Recomendation: grapes')
    elif prediction == 14:
        print('Crop Recomendation: watermelon')
    elif prediction == 1:
        print('Crop Recomendation: apple')
    elif prediction == 11:
        print('Crop Recomendation: orange')
    elif prediction == 5:
        print('Crop Recomendation: cotton')
    elif prediction == 4:
        print('Crop Recomendation: coffee')
    # print("Accuracy Score:", accuracy)
    print("=" * 60)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    path_data = "/home/kemal/Documents/project/dataset/Crop_recommendation.csv"
    path_current = input("input the data path: ")
    current_data = pd.read_csv(path_current) #"/home/kemal/Documents/project/deploy/test.csv"

    data = readFile(path_data)
    X, y = xySplitter(data)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = trainTest(X, y)

    model = LogisticRegression()
    trained_model = modeling(model, X_train, y_train)
    
    print("=" * 60)
    print(current_data)
    predictData(trained_model, current_data)
