from train_model import train_model
from test_model import test_model


print('\nDecisionTree Regression:')
train_model(file_name="train.csv",model_name='DecisionTree Regression')
test_model(file_name='new_input.csv',model_name='DecisionTree Regression')

print('\nKNeighbours Regression:')
train_model(file_name="train.csv",model_name='KNeighbours Regression')
test_model(file_name='new_input.csv',model_name='KNeighbours Regression')

print('\nGradientBoosting Regression:')
train_model(file_name="train.csv",model_name='GradientBoosting Regression')
test_model(file_name='new_input.csv',model_name='GradientBoosting Regression')

print('\nRandomForest Regression:')
train_model(file_name="train.csv",model_name='RandomForest Regression')
test_model(file_name='new_input.csv',model_name='RandomForest Regression')

print('\nXBG Regression:')
train_model(file_name="train.csv",model_name='XBG Regression')
test_model(file_name='new_input.csv',model_name='XBG Regression')

print('\nNeuralNetwork Regression:')
train_model(file_name="train.csv",model_name='NeuralNetwork Regression')
test_model(file_name='new_input.csv',model_name='NeuralNetwork Regression')