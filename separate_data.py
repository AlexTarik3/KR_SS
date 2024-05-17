import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/Omen/Desktop/ІТСС/RGR/data/Price_P.csv')

train, test = train_test_split(df, train_size=0.9, test_size=0.1, stratify=df['price'])

train.to_csv('C:/Users/Omen/Desktop/ІТСС/RGR/data/train.csv', index=False)
test.to_csv('C:/Users/Omen/Desktop/ІТСС/RGR/data/new_input.csv', index=False)