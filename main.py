import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('parkinsons.csv')

# Select features and target
X = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation accuracy:", accuracy)

# Save model
joblib.dump(model, 'my_model.joblib')
