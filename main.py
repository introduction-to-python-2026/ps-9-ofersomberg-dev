import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# Load data
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
print(df.columns.to_list())

selected_features = ['RPDE','PPE']
X = df[selected_fetures]
y = df['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

DTC = DecisionTreeClassifier(max_depth = 3)
DTC.fit(X_train, y_train)

y_pred = DTC.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Save model
joblib.dump(model, 'my_model.joblib')
