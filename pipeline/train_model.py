import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from preprocessing import preprocess_train_data

def train_model(file_name: str = 'train.csv', model_name: str = 'xgboost'):
    # loading data
    data = pd.read_csv('C:/Users/Omen/Desktop/ІТСС/RGR/data/' + file_name)

    # preprocessing data
    data = preprocess_train_data(data)

    # split data
    X = data.drop(columns=['price'])
    Y = data['price']

    # models
    models = {
        'DecisionTree Regression': DecisionTreeRegressor(),
        'KNeighbours Regression': KNeighborsRegressor(),
        'GradientBoosting Regression': GradientBoostingRegressor(verbose=0),
        'RandomForest Regression': RandomForestRegressor(),
        'XBG Regression': XGBRegressor(verbose = 0),
        'NeuralNetwork Regression': MLPRegressor(verbose=0)
    }

    # training model
    model = models[model_name]
    model.fit(X, Y)

    # saving model
    with open(f'C:/Users/Omen/Desktop/ІТСС/RGR/models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)