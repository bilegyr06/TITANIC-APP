import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
# We use a direct link to the Titanic CSV for easy access
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Data Preprocessing (Cleaning)
# Select features: Pclass (1st/2nd/3rd), Sex, Age, SibSp (Siblings), Parch (Parents), Fare
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Fill missing Age values with the median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Fare values with the median fare
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Convert 'Sex' text to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 3. Define Features (X) and Target (y)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save the model as 'titanicmodel.pkl'
with open('titanicmodel.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Success: Model trained and saved as 'titanicmodel.pkl'")