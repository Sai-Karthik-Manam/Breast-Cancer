import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Loading the dataset
df = pd.read_csv("Cancer.csv")

# Preparing features and target
X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
y = df['diagnosis']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Training the Random Forest model
rf_model = RandomForestClassifier(random_state=2529)
rf_model.fit(X_train, y_train)

# Saving the trained model to rf_model.pkl
joblib.dump(rf_model, "rf_model.pkl")

print(" Model trained and saved as rf_model.pkl")
