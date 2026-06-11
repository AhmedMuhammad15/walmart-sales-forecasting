import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Walmart Sales Analytics", layout="wide")

st.title("Walmart Retail Sales Analytics & Forecasting Dashboard")
st.write("Project: *Predictive Analysis and Time-Series Forecasting of Walmart Retail Sales*")

# File Uploader in Sidebar
uploaded_file = st.sidebar.file_uploader("Upload Walmart CSV File", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # FIX: Column names ke aage peeche se spaces khatam karna
    df.columns = df.columns.str.strip()
    
    # FIX: Agar 'date' small letters mai ho toh usko standard 'Date' banana
    if 'date' in df.columns and 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    if 'weekly_sales' in df.columns and 'Weekly_Sales' not in df.columns:
        df.rename(columns={'weekly_sales': 'Weekly_Sales'}, inplace=True)
        
    # Walmart ke zaroori columns verification
    required_columns = ['Store', 'Dept', 'Date', 'Weekly_Sales']
    check_passed = all(col in df.columns for col in required_columns)
    
    if not check_passed:
        st.error("❌ **Invalid Dataset Format!**")
        st.write(f"SRE/IDS Project Scope Check Failed. Aapki file ke columns yeh hain: `{list(df.columns)}`")
        st.write("Hume strictly Walmart ke columns chahiye: `Store`, `Dept`, `Date`, `Weekly_Sales`")
    else:
        st.success("✅ Walmart Dataset Loaded & Standardized Successfully!")
        
        # Preprocessing (Date handling safely)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows where date formatting failed
        df = df.dropna(subset=['Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
        
        # Tabs Layout
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📉 Macroeconomic Analysis", "🔮 Real-Time XGBoost Forecast"])
        
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total Rows / Records", value=f"{df.shape[0]:,}")
            with col2:
                st.metric(label="Total Unique Stores", value=df['Store'].nunique())

        with tab2:
            st.subheader("Macroeconomic Factors Impact (Justification Page)")
            st.write("Visual proof for Sir showing how external economic metrics influence purchasing power and trends.")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Correlation Matrix (CPI, Fuel, Unemployment vs Sales)**")
                fig, ax = plt.subplots(figsize=(8, 5))
                macro_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
                existing_cols = [c for c in macro_cols if c in df.columns]
                sns.heatmap(df[existing_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
                st.caption("Notice the correlation weights. Non-zero values prove these metrics influence sales trends.")
                
            with col_chart2:
                st.write("**Sales Distribution: Holiday Weeks vs Regular Weeks**")
                fig, ax = plt.subplots(figsize=(8, 5))
                # Case insensitive check for IsHoliday
                holiday_col = 'IsHoliday' if 'IsHoliday' in df.columns else ('isholiday' if 'isholiday' in df.columns else None)
                if holiday_col:
                    sns.boxplot(data=df, x=holiday_col, y='Weekly_Sales', palette="Set2", ax=ax)
                    plt.yscale('log')
                    st.pyplot(fig)
                else:
                    st.warning("IsHoliday column not found for plotting.")

        with tab3:
            st.subheader("Dynamic XGBoost Forecasting Engine")
            st.write("Select a specific Store to train the algorithm and visualize real-time dynamic fitting.")
            
            selected_store = st.selectbox("Select Store for Analysis", sorted(df['Store'].unique()))
            
            if st.button("🚀 Run Dynamic Training & Forecast"):
                with st.spinner("Training XGBoost Regressor on the uploaded file structure... This might take a moment."):
                    
                    # Feature Engineering for Model
                    ml_df = df.copy()
                    le = LabelEncoder()
                    
                    if 'Type' in ml_df.columns:
                        ml_df['Type'] = le.fit_transform(ml_df['Type'].astype(str))
                    if 'type' in ml_df.columns:
                        ml_df['type'] = le.fit_transform(ml_df['type'].astype(str))
                        
                    holiday_col = 'IsHoliday' if 'IsHoliday' in ml_df.columns else ('isholiday' if 'isholiday' in ml_df.columns else None)
                    if holiday_col:
                        ml_df[holiday_col] = ml_df[holiday_col].astype(int)
                    
                    # Target features list matching both lower/upper cases safely
                    features = ['Store', 'store', 'Dept', 'dept', 'IsHoliday', 'isholiday', 'Size', 'size', 
                                'Temperature', 'temperature', 'Fuel_Price', 'fuel_price', 'CPI', 'cpi', 
                                'Unemployment', 'unemployment', 'Year', 'Month', 'Week']
                    
                    actual_features = [f for f in features if f in ml_df.columns]
                    
                    X = ml_df[actual_features]
                    y = ml_df['Weekly_Sales']
                    
                    # Fast Training Setup (Optimized for laptop performance)
                    model = XGBRegressor(n_estimators=30, max_depth=5, random_state=42)
                    model.fit(X, y)
                    
                    st.success("Model Training Complete!")
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        st.write("**Feature Importance Chart**")
                        fig, ax = plt.subplots()
                        feat_imp = pd.Series(model.feature_importances_, index=actual_features).sort_values(ascending=True)
                        feat_imp.plot(kind='barh', color='darkorange', ax=ax)
                        st.pyplot(fig)
                        st.caption("Proof that CPI and Fuel Price have predictive weight in machine learning splits.")
                    
                    with col_res2:
                        st.write(f"**Actual vs Predicted Trend Line for Store {selected_store}**")
                        store_col = 'Store' if 'Store' in ml_df.columns else 'store'
                        store_mask = ml_df[store_col] == selected_store
                        store_data = ml_df[store_mask].sort_values('Date')
                        
                        if not store_data.empty:
                            predictions = model.predict(store_data[actual_features])
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.plot(store_data['Date'], store_data['Weekly_Sales'], label='Actual Sales', alpha=0.5, color='blue')
                            ax.plot(store_data['Date'], predictions, label='XGBoost Predicted', alpha=0.8, color='red', linestyle='--')
                            plt.xticks(rotation=45)
                            plt.legend()
                            st.pyplot(fig)
                        else:
                            st.warning("Not enough continuous points to display standard trend.")
else:
    st.info("💡 Awaiting Walmart CSV file upload from the sidebar to activate the forecasting dashboard.")