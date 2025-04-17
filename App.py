import pandas as pd
import numpy as np
import joblib  # Save & load models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv('Pred-Main.csv')

# Drop unnecessary columns
df = df.drop(['Product ID', 'UDI'], axis=1)

# Remove extra space in column
df.columns = df.columns.str.strip()

# Convert temperature from Kelvin to Celsius
df['Process temperature [C]'] = df['Process temperature [K]'] - 273.15
df['Air temperature [C]'] = df['Air temperature [K]'] - 273.15
df.drop(['Process temperature [K]', 'Air temperature [K]'], axis=1, inplace=True)

# Encode categorical column
label_encoder = LabelEncoder()
df['M_Type'] = label_encoder.fit_transform(df['Type'])
df.drop(['Type'], axis=1, inplace=True)

# Features & Target
X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# Oversampling for class imbalance
ros = RandomOverSampler(sampling_strategy='all', random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Define ML pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

# Train pipeline
pipeline.fit(X_resampled, y_resampled)

# Save pipeline
joblib.dump(pipeline, "ml_pipeline.pkl")
print("Model pipeline saved as 'ml_pipeline.pkl'")

# ------------------- Get User Input -------------------

def get_user_input():
    """Collects user input for prediction"""
    process_temp = float(input("Enter Process Temperature [C]: "))
    air_temp = float(input("Enter Air Temperature [C]: "))
    vibration = float(input("Enter Vibration Levels: "))
    operational_hours = float(input("Enter Operational Hours: "))
    rpm = float(input("Enter Rotational Speed [rpm]: "))
    torque = float(input("Enter Torque [Nm]: "))
    m_type = int(input("Enter M_Type (0 or 1): "))

    return pd.DataFrame([[rpm, torque, vibration, operational_hours, process_temp, air_temp, m_type]],
                        columns=["Rotational speed [rpm]", "Torque [Nm]", "Vibration Levels",
                                 "Operational Hours", "Process temperature [C]", "Air temperature [C]", "M_Type"])

# Load the trained pipeline
pipeline = joblib.load("ml_pipeline.pkl")

# Get user input and predict
user_input = get_user_input()
prediction = pipeline.predict(user_input)[0]

print(f" Predicted Failure Type: {prediction}")