# slm-predictive-maintenance-iot

# SmartEdgePM: Predictive Maintenance using Small Language Models (SLM) and IoT Data

This project leverages IoT sensor data and integrates it with a Small Language Model (SLM) to perform predictive maintenance. It detects potential machine failures based on real-time telemetry and provides intelligent, AI-driven insights for prevention and optimization.

##  Features
-  Predicts machine failure types using ML models (KNN)
-  Uses a Small Language Model (SLM) like Phi-3 for generating actionable maintenance insights
-  Designed for IoT-based edge deployment
-  Real-time human-like explanations of system failures

## Dataset
   Engine Failure dataset (Kaggle / Custom):
   Includes telemetry like:

-  Rotational speed [rpm]
-  Torque [Nm]
-  Vibration levels
-  Operational hours
-  Process & Air Temperature
-  Machine Type

## LLM Integration
Using Phi-3 for:
-  Explaining predicted failure modes
-  Providing preventive maintenance tips
-  Generating maintenance reports on the fly
## Future Improvements
-  Integration with MQTT/Edge devices
-  Real-time dashboard with alerts
