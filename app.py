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

# Title
st.title("ML Regression Web App")

# Function for Data Preprocessing
def preprocess_data(df):
    # Handling missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Converting categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    
    return df

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    st.write("### Data Preview:", df.head())
    
    # Feature selection
    features = st.multiselect("Select Feature Columns", df.columns.tolist())
    target = st.selectbox("Select Target Column", df.columns.tolist())
    
    if features and target:
        X = df[features]
        y = df[target]
        
        # Scale target variable (optional, useful for large magnitude values)
        scaler = StandardScaler()
        y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        scaled_target_mean = np.mean(y)
        target_scale = np.std(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model selection
        model_options = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", 
                         "Random Forest Regressor", "Extra Trees Regressor", "Support Vector Regressor", 
                         "Gradient Boosting Regressor"]
        selected_models = st.multiselect("Select Models to Compare", model_options, default=model_options[:3])
        
        alpha = st.slider("Select Alpha (for Ridge/Lasso)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        n_estimators = st.slider("Select n_estimators (for ensemble models)", min_value=10, max_value=200, value=100, step=10)
        
        results = []
        
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
        
        # Display results table
        results_df = pd.DataFrame(results)
        st.write("### Model Comparison Table")
        st.dataframe(results_df)
        
        # Calculate recommended ranges based on dataset
        mae_range = (scaled_target_mean * 0.1, scaled_target_mean * 0.3)
        mse_range = (target_scale * 0.1, target_scale * 0.3)
        rmse_range = (target_scale * 0.1, target_scale * 0.3)
        r2_good = "> 0.8 (Closer to 1 is better)"
        
        # # Display performance guideline
        # st.write("### Performance Guidelines")
        # st.write(f"- A good **MAE** should be close to sclaed target mean:  {scaled_target_mean} ")
        # st.write(f"- A good **MSE** should be between {mse_range[0]:.4f} and {mse_range[1]:.4f}")
        # st.write(f"- A good **RMSE** should be between {rmse_range[0]:.4f} and {rmse_range[1]:.4f}")
        # st.write(f"- A good **R² Score** should be {r2_good}")
        
        # Allow user to download comparison table
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Model Comparison Table", data=csv, file_name="model_comparison.csv", mime="text/csv")
        
        # Allow user to download full dataset with predictions
        dataset_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Dataset with Predictions", data=dataset_csv, file_name="dataset_with_predictions.csv", mime="text/csv")
