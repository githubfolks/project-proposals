
# Election Prediction ML Model - Starter Notebook

## 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

## 2. Load Dataset
# Replace with your own CSV file containing election data
# Example columns: ['year', 'state', 'constituency', 'party', 'voter_turnout', 'development_index', 'is_incumbent', 'winning_party']
df = pd.read_csv('election_data.csv')
df.head()

## 3. Data Preprocessing
# Example: Drop rows with missing values
df.dropna(inplace=True)

# Feature and label selection
X = df[['year', 'voter_turnout', 'development_index', 'is_incumbent', 'party']]
y = df['winning_party']

# Define categorical and numerical features
categorical_features = ['party']
numerical_features = ['year', 'voter_turnout', 'development_index', 'is_incumbent']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(), categorical_features)
])

## 4. Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

## 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## 6. Train Model
model.fit(X_train, y_train)

## 7. Evaluation
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

## 8. Feature Importance (Optional)
# Only works for models that support it like RandomForest
classifier = model.named_steps['classifier']
feature_names = model.named_steps['preprocessor'].transformers_[0][2] +                 list(model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features))
importances = classifier.feature_importances_
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.show()
