import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('parkinsons.csv')

# Select features and label
selected_features = ['PPE', 'MDVP:Fo()Hz']
x = df[selected_features]
y = df['status']

# Scale data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
