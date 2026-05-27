import pandas as pd
import pickle
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
data = pl.scan_csv('data/emails.csv')

x = data.drop(['Email No.', 'Prediction']).collect().to_pandas()
y = data.select('Prediction').collect().to_pandas()

X_train , X_test , Y_train , Y_test = train_test_split(x, y , test_size=0.2 , random_state=42)

model = MultinomialNB()
model.fit(X_train, Y_train)

print("Model trained successfully")

with open('model/model.pkl','wb') as file:
    pickle.dump(model,file)

print("Model saved successfully")
accuracy = model.score(X_test, Y_test)
percentage = accuracy * 100
print(f"Accuracy: {percentage:.2f}%")

