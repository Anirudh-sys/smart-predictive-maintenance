import pandas as pd
import numpy as np
import joblib
import requests
from langchain_ollama import OllamaLLM

# Save & load models
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
'''
# Function to generate insights using Ollama with Phi-3
def get_ollama_insights(prediction, input_data):
    """
    Get insights from Ollama Phi-3 model based on ML prediction
    
    Args:
        prediction: Predicted failure type
        input_data: DataFrame with input parameters
    
    Returns:
        Generated insights as string
    """
    # Format the input data
    input_dict = input_data.iloc[0].to_dict()
    
    # Create a prompt for Phi-3
    prompt = f"""
    You are a machine failure analysis expert. Analyze this prediction:
    
    MACHINE PARAMETERS:
    - Rotational speed: {input_dict['Rotational speed [rpm]']} rpm
    - Torque: {input_dict['Torque [Nm]']} Nm
    - Vibration: {input_dict['Vibration Levels']}
    - Hours: {input_dict['Operational Hours']}
    - Process temp: {input_dict['Process temperature [C]']} °C
    - Air temp: {input_dict['Air temperature [C]']} °C
    - Machine type: {input_dict['M_Type']}
    
    PREDICTION: {prediction}
    
    Provide a concise analysis including:
    1. Most likely causes based on these parameters
    2. Recommended maintenance actions
    3. Preventative measures
    """
    
    # Call Ollama API with increased timeout
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'phi3:mini ',
                'prompt': prompt,
                'stream': False
            },
            timeout=180  # 3 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error from Ollama API: {response.status_code}"
            
    except Exception as e:
        return f"Failed to connect to Ollama: {str(e)}"
'''


# Initialize Ollama LangChain model
llm = OllamaLLM(model="phi3:mini")  

def get_failure_insights(prediction, input_data):
    """Generate maintenance insights using LangChain with Ollama"""

    prompt = f"""
    The machine has been classified with a failure type of '{prediction}'.
    Provide a short explanation and 3 quick maintenance steps within 60 words.
    """

    try:
        response = llm.invoke(prompt)
        return response.strip() if response else "⚠️ No response received from Ollama."
    
    except Exception as e:
        return f"⚠️ Exception: {str(e)}"

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
    m_type = int(input("Enter M_Type (0=L, 1=M, 2=H): "))
    
    return pd.DataFrame([[rpm, torque, vibration, operational_hours, process_temp, air_temp, m_type]],
                        columns=["Rotational speed [rpm]", "Torque [Nm]", "Vibration Levels", 
                                 "Operational Hours", "Process temperature [C]", "Air temperature [C]", "M_Type"])

# Load the trained pipeline
pipeline = joblib.load("ml_pipeline.pkl")


# ------------------- Get Prediction -------------------

# Get user input and predict failure type
user_input = get_user_input()
prediction = pipeline.predict(user_input)[0]

print(f"\n Predicted Failure Type: {prediction}")

# Define failure types that require insights
failure_types = ["Power Failure", "Tool Wear Failure", "Overstrain Failure"]  

# ------------------- Generate Insights Only for Failures -------------------

if prediction in failure_types:
    print("\nGenerating insights with Phi-3 Mini...\n")
    insights = get_failure_insights(prediction,user_input)
    print("\n===== MACHINE FAILURE INSIGHTS (PHI-3 Mini) =====")
    print(insights)
    print("=============================================")

    # Save insights to a file
    with open("failure_insights.txt", "w", encoding="utf-8") as f:
        f.write(f"INSIGHTS:\n{insights}")

    print("Insights saved to 'failure_insights.txt'")
else:
    print("\n No failure detected. No insights required.")

'''
# Print the prediction
# print(f"\n🔍 Predicted Failure Type: {prediction}")

# Generate insights with Ollama Phi-3
print("Generating insights with Phi-3...")
insights = get_failure_insights(prediction, user_input)

# Display insights
print("\n===== MACHINE FAILURE INSIGHTS (PHI-3) =====")
print(insights)
print("=============================================")

# Save to file
with open("failure_insights.txt", "w", encoding="utf-8") as f:
    f.write(f"INSIGHTS:\n{insights}")

print("\nInsights saved to 'failure_insights.txt'")'
'''