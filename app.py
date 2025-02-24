import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="ML Regressor Hub", page_icon="📊", layout="wide")


# Title with styling
st.markdown("""
    <h1 style='text-align: center; color: #3366ff;'>ML Regressor Hub</h1>
    <hr style='border: 1px solid #ccc;'>
""", unsafe_allow_html=True)
with st.expander("📌 When Not to Use Regression Models"):
            st.markdown("""
            - When dealing with **classification problems** (e.g., predicting categories instead of continuous values).
            - If the dataset has **highly non-linear relationships** that regression cannot capture well.
            - When working with **unstructured data** like images, text, or audio.
            - If the dataset has **high-dimensional sparse features**, tree-based models or deep learning might work better.
            - For **time-series forecasting**, as regression models don’t inherently consider sequential dependencies.
            """, unsafe_allow_html=True)
# Function for Data Preprocessing
def preprocess_data(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    return df

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())
    
    # Feature selection
    st.sidebar.header("Feature Selection")
    features = st.sidebar.multiselect("Select Feature Columns", df.columns.tolist())
    target = st.sidebar.selectbox("Select Target Column", df.columns.tolist())
    
    if features and target:
        X = df[features]
        y = df[target]
        
        scaler = StandardScaler()
        y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        st.sidebar.header("Model Selection")
        model_options = [
            "Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", 
            "Random Forest Regressor", "Extra Trees Regressor", "Support Vector Regressor", 
            "Gradient Boosting Regressor"
        ]
        selected_models = st.sidebar.multiselect("Select Models to Compare", model_options, default=model_options[:3])
        
        alpha = st.sidebar.slider("Alpha (for Ridge/Lasso)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        n_estimators = st.sidebar.slider("n_estimators (for ensemble models)", min_value=10, max_value=200, value=100, step=10)
        
        results = []
        
        with st.spinner("Training models..."):
            for model_choice in selected_models:
                model = None
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                elif model_choice == "Lasso Regression":
                    model = Lasso(alpha=alpha)
                elif model_choice == "Decision Tree Regressor":
                    model = DecisionTreeRegressor()
                elif model_choice == "Random Forest Regressor":
                    model = RandomForestRegressor(n_estimators=n_estimators)
                elif model_choice == "Extra Trees Regressor":
                    model = ExtraTreesRegressor(n_estimators=n_estimators)
                elif model_choice == "Support Vector Regressor":
                    model = SVR()
                elif model_choice == "Gradient Boosting Regressor":
                    model = GradientBoostingRegressor(n_estimators=n_estimators)
                
                if model is not None:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    df[f"Prediction_{model_choice}"] = scaler.inverse_transform(model.predict(X).reshape(-1, 1)).flatten()
                    
                    results.append({
                        "Model": model_choice,
                        "MAE": mean_absolute_error(y_test, y_pred),
                        "MSE": mean_squared_error(y_test, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                        "R² Score": r2_score(y_test, y_pred)
                    })
        
        results_df = pd.DataFrame(results)
        
        st.subheader("📊 Model Comparison Table")
        st.dataframe(results_df.style.highlight_min(axis=0, color='lightcoral').highlight_max(axis=0, color='lightgreen'))
        
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Model Comparison Table", data=csv, file_name="model_comparison.csv", mime="text/csv")
        
        dataset_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Dataset with Predictions", data=dataset_csv, file_name="dataset_with_predictions.csv", mime="text/csv")