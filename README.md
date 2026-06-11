# 🛒 Walmart Retail Sales Analytics & Forecasting

This is a complete Data Science and Machine Learning project built with Python and Streamlit. It focuses on Predictive Analysis and Time-Series Forecasting of Walmart Retail Sales using an XGBoost Regressor.

## 🌟 Key Features
- **📋 Data Overview:** Standardizes the dataset automatically and provides high-level metrics (total records, unique stores).
- **📉 Macroeconomic Analysis:** Visualizes how external economic factors (CPI, Fuel Price, Unemployment) influence retail sales through correlation matrices and trend charts.
- **🔮 Real-Time Forecasting Engine:** A dynamic XGBoost model that trains on the fly based on the selected store. It provides feature importance charts and compares actual vs. predicted sales trends.

## 🛠️ Technologies Used
- Python
- Streamlit (for the interactive web dashboard)
- XGBoost (for Machine Learning forecasting)
- Pandas & NumPy (for data manipulation)
- Matplotlib & Seaborn (for data visualization)
- Scikit-Learn (for data preprocessing)

## 🚀 How to Run Locally
1. Clone this repository:
   ```bash
   git clone (https://github.com/AhmedMuhammad15/walmart-sales-forecasting.git)
Navigate into the folder and install the required packages:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
Upload your Walmart csv dataset in the sidebar to activate the dashboard!

🌐 Live Demo
You can try out the live application here: https://walmart-sales-forecasting-analytics.streamlit.app/