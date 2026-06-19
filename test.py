import pandas as pd

df = pd.read_csv("placement-dataset.csv")

print(df.head())
print(df.columns)
print(df.shape)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("placement-dataset.csv")

# Remove useless column
df = df.drop(columns=['Unnamed: 0'])

# Features and target
X = df[['cgpa', 'iq']]
y = df['placement']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Save model
pickle.dump(
    model,
    open("model.pkl", "wb")
)

print("Model trained successfully!")

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

