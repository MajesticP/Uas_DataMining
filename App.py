import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import boxcox
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt


st.set_page_config(page_title="Reggresion", layout="wide")


st.sidebar.title("Choose a Task")
task = st.sidebar.radio(
    "Select a task:",
    [
        "About App",
        "Support Vector Analysis",  
        "Decision Tree Analysis",
        "Random Forest Regression", 
        "Ridge Regression",  
        "Model Performance Comparison",  
    ],
)


if "data" not in st.session_state:
    st.session_state["data"] = pd.read_csv("Regression.csv")  


def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    return st.session_state["data"]  


if task == "About App":
    st.write("### About Me")
    st.write('**Nama**: Muhammad Saddam Ulfikri')
    st.write('**NIM**: 211220029')
    st.write("### About This App")
    st.write("""
    Welcome to the **Model Perfomance Comparison APP**! This app allows you to:
    - Perform **SVM Support Vector Analysis**.
    - Analyze data using **Decision Tree Regression**.
    - Perform **Random Forest Regression Analysis**.
    - Perform **Ridge Regression Analysis**.
    - Compare the performance of multiple models.
    - Visualize and filter your dataset with ease.

    You can use the default dataset (`Regression.csv`) or upload your own CSV file. Just make sure it's the same **goddamn** file (╬▔皿▔)╯  nothing more, nothing less! Get ready to experience an **EXTRAORDINARY** analysis journey. Your data is going to be **REVOLUTIONIZED**! MAYBE (○｀ 3′○)
    """)


    st.write("### Default Dataset Preview:")
    st.write(st.session_state["data"])

 
    st.write("### Upload Your Dataset")
    file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if file:
        st.session_state["data"] = load_data(file)
        st.write("### Uploaded Dataset Preview:")
        st.write(st.session_state["data"])

elif task == "Support Vector Analysis":
    st.write("### Support Vector Analysis")
    data = st.session_state.get("data")

    if data is None:
        st.warning("Please upload a dataset in the 'About App' section.")
    else:
        # Sidebar filters
        st.sidebar.title("Filter Options")
        filtered_data = data.copy()
        if "age" in data.columns:
            age_range = st.sidebar.slider(
                "Select Age Range",
                int(data["age"].min()),
                int(data["age"].max()),
                (int(data["age"].min()), int(data["age"].max()))
            )
            filtered_data = filtered_data[(filtered_data["age"] >= age_range[0]) & (filtered_data["age"] <= age_range[1])]

        if "bmi" in data.columns:
            bmi_range = st.sidebar.slider(
                "Select BMI Range",
                float(data["bmi"].min()),
                float(data["bmi"].max()),
                (float(data["bmi"].min()), float(data["bmi"].max()))
            )
            filtered_data = filtered_data[(filtered_data["bmi"] >= bmi_range[0]) & (filtered_data["bmi"] <= bmi_range[1])]

        if "children" in data.columns:
            children_filter = st.sidebar.multiselect(
                "Select Number of Children",
                options=sorted(data["children"].unique())
            )
            if children_filter:
                filtered_data = filtered_data[filtered_data["children"].isin(children_filter)]

        if "smoker" in data.columns:
            smoker_filter = st.sidebar.radio("Include Smokers?", ("Yes", "No", "Both"))
            if smoker_filter == "Yes":
                filtered_data = filtered_data[filtered_data["smoker"] == "yes"]
            elif smoker_filter == "No":
                filtered_data = filtered_data[filtered_data["smoker"] == "no"]

        if "region" in data.columns:
            region_filter = st.sidebar.multiselect(
                "Select Regions",
                options=sorted(data["region"].unique())
            )
            if region_filter:
                filtered_data = filtered_data[filtered_data["region"].isin(region_filter)]

        st.write("### Filtered Data Preview:")
        st.write(filtered_data)

        unused_data = data[~data.index.isin(filtered_data.index)]
        if not unused_data.empty:
            st.write("### Unused Data Preview:")
            st.write(unused_data)
        categorical_features = ["sex", "smoker", "region"]
        numerical_features = ["age", "bmi", "children"]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(drop="first"), categorical_features),
            ]
        )
        X = filtered_data.drop("charges", axis=1)
        y = filtered_data["charges"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", SVR(kernel="rbf"))
        ])


        st.write("### Hyperparameter Tuning")
        param_grid = {
            "model__C": [0.1, 1, 10, 100],
            "model__epsilon": [0.1, 0.2, 0.5, 1],
            "model__gamma": ["scale", "auto"],
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        st.write("Best Hyperparameters:", grid_search.best_params_)

        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance:")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R²): {r2:.2f}")

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual Charges")
        ax.set_ylabel("Predicted Charges")
        ax.set_title("Actual vs Predicted Charges")
        st.pyplot(fig)

elif task == "Decision Tree Analysis":
    st.write("### Decision Tree Regression Analysis")
    data = st.session_state.get("data")
    
    if data is None:
        st.warning("Please upload a dataset in the 'About App' section.")
    else:
        
        st.sidebar.title("Filter Options")
        if "age" in data.columns:
            age_range = st.sidebar.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (int(data["age"].min()), int(data["age"].max())))
            data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

        if "bmi" in data.columns:
            bmi_range = st.sidebar.slider("Select BMI Range", float(data["bmi"].min()), float(data["bmi"].max()), (float(data["bmi"].min()), float(data["bmi"].max())))
            data = data[(data["bmi"] >= bmi_range[0]) & (data["bmi"] <= bmi_range[1])]

        if "children" in data.columns:
            children_filter = st.sidebar.multiselect("Select Number of Children", options=sorted(data["children"].unique()))
            if children_filter:
                data = data[data["children"].isin(children_filter)]

        if "smoker" in data.columns:
            smoker_filter = st.sidebar.radio("Include Smokers?", ("Yes", "No", "Both"))
            if smoker_filter == "Yes":
                data = data[data["smoker"] == "yes"]
            elif smoker_filter == "No":
                data = data[data["smoker"] == "no"]

        if "region" in data.columns:
            region_filter = st.sidebar.multiselect("Select Regions", options=sorted(data["region"].unique()))
            if region_filter:
                data = data[data["region"].isin(region_filter)]

        st.write("### Filtered Data Preview:")
        st.write(data)

        
        data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

        
        X = data.drop("charges", axis=1)
        y = data["charges"]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
        max_depth = st.sidebar.slider("Select Tree Depth", 1, 20, 5) 
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

      
        y_pred = model.predict(X_test)

      
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)  
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance:")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"Mean Absolute Error: {mae:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, feature_names=X.columns, filled=True, ax=ax)
        plt.title("Decision Tree")
        st.pyplot(fig)


elif task == "Random Forest Regression":
    st.write("### Random Forest Regression")
    data = st.session_state.get("data")
    
    if data is None:
        st.warning("Please upload a dataset in the 'About App' section.")
    else:
        
        st.sidebar.title("Filter Options")
        if "age" in data.columns:
            age_range = st.sidebar.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (int(data["age"].min()), int(data["age"].max())))
            data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

        if "bmi" in data.columns:
            bmi_range = st.sidebar.slider("Select BMI Range", float(data["bmi"].min()), float(data["bmi"].max()), (float(data["bmi"].min()), float(data["bmi"].max())))
            data = data[(data["bmi"] >= bmi_range[0]) & (data["bmi"] <= bmi_range[1])]

        if "children" in data.columns:
            children_filter = st.sidebar.multiselect("Select Number of Children", options=sorted(data["children"].unique()))
            if children_filter:
                data = data[data["children"].isin(children_filter)]

        if "smoker" in data.columns:
            smoker_filter = st.sidebar.radio("Include Smokers?", ("Yes", "No", "Both"))
            if smoker_filter == "Yes":
                data = data[data["smoker"] == "yes"]
            elif smoker_filter == "No":
                data = data[data["smoker"] == "no"]

        if "region" in data.columns:
            region_filter = st.sidebar.multiselect("Select Regions", options=sorted(data["region"].unique()))
            if region_filter:
                data = data[data["region"].isin(region_filter)]

        st.write("### Filtered Data Preview:")
        st.write(data)

        
        data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

        
        X = data.drop("charges", axis=1)
        y = data["charges"]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        
        y_pred = model.predict(X_test)

       
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance:")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

       
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Charges")
        st.pyplot(fig)


elif task == "Ridge Regression":
    st.write("### Ridge Regression")
    data = st.session_state.get("data")
    
    if data is None:
        st.warning("Please upload a dataset in the 'About App' section.")
    else:
        
        st.sidebar.title("Filter Options")
        if "age" in data.columns:
            age_range = st.sidebar.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (int(data["age"].min()), int(data["age"].max())))
            data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

        if "bmi" in data.columns:
            bmi_range = st.sidebar.slider("Select BMI Range", float(data["bmi"].min()), float(data["bmi"].max()), (float(data["bmi"].min()), float(data["bmi"].max())))
            data = data[(data["bmi"] >= bmi_range[0]) & (data["bmi"] <= bmi_range[1])]

        if "children" in data.columns:
            children_filter = st.sidebar.multiselect("Select Number of Children", options=sorted(data["children"].unique()))
            if children_filter:
                data = data[data["children"].isin(children_filter)]

        if "smoker" in data.columns:
            smoker_filter = st.sidebar.radio("Include Smokers?", ("Yes", "No", "Both"))
            if smoker_filter == "Yes":
                data = data[data["smoker"] == "yes"]
            elif smoker_filter == "No":
                data = data[data["smoker"] == "no"]

        if "region" in data.columns:
            region_filter = st.sidebar.multiselect("Select Regions", options=sorted(data["region"].unique()))
            if region_filter:
                data = data[data["region"].isin(region_filter)]

        st.write("### Filtered Data Preview:")
        st.write(data)

        
        data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

        
        X = data.drop("charges", axis=1)
        y = data["charges"]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        
        y_pred = model.predict(X_test)

       
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance:")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

       
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted Charges")
        st.pyplot(fig)


elif task == "Model Performance Comparison":
    st.write("### Model Performance Comparison")
    data = st.session_state.get("data")
    
    if data is None:
        st.warning("Please upload a dataset in the 'About App' section.")
    else:
        
        st.sidebar.title("Filter Options")
        if "age" in data.columns:
            age_range = st.sidebar.slider("Select Age Range", int(data["age"].min()), int(data["age"].max()), (int(data["age"].min()), int(data["age"].max())))
            data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

        if "bmi" in data.columns:
            bmi_range = st.sidebar.slider("Select BMI Range", float(data["bmi"].min()), float(data["bmi"].max()), (float(data["bmi"].min()), float(data["bmi"].max())))
            data = data[(data["bmi"] >= bmi_range[0]) & (data["bmi"] <= bmi_range[1])]

        if "children" in data.columns:
            children_filter = st.sidebar.multiselect("Select Number of Children", options=sorted(data["children"].unique()))
            if children_filter:
                data = data[data["children"].isin(children_filter)]

        if "smoker" in data.columns:
            smoker_filter = st.sidebar.radio("Include Smokers?", ("Yes", "No", "Both"))
            if smoker_filter == "Yes":
                data = data[data["smoker"] == "yes"]
            elif smoker_filter == "No":
                data = data[data["smoker"] == "no"]

        if "region" in data.columns:
            region_filter = st.sidebar.multiselect("Select Regions", options=sorted(data["region"].unique()))
            if region_filter:
                data = data[data["region"].isin(region_filter)]

        st.write("### Filtered Data Preview:")
        st.write(data)

       
        data = pd.get_dummies(data, columns=["sex", "smoker", "region"], drop_first=True)

       
        X = data.drop("charges", axis=1)
        y = data["charges"]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
        models = {
            "SVM": SVR(kernel='rbf'),
            "Decision Tree": DecisionTreeRegressor(max_depth=5),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Ridge Regression": Ridge(alpha=1.0)
        }

        
        model_results = {}

        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model_results[model_name] = {"MSE": mse, "R2": r2}

        st.write("### Model Comparison Table")
        comparison_df = pd.DataFrame(model_results).T
        comparison_df.sort_values("R2", ascending=False, inplace=True)
        st.write(comparison_df)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.barplot(x=comparison_df.index, y=comparison_df["MSE"], ax=axes[0], palette='Blues_d')
        axes[0].set_title("Mean Squared Error (MSE) Comparison")
        axes[0].set_ylabel("MSE")
        axes[0].set_xlabel("Model")

        sns.barplot(x=comparison_df.index, y=comparison_df["R2"], ax=axes[1], palette='Greens_d')
        axes[1].set_title("R2 Score Comparison")
        axes[1].set_ylabel("R2 Score")
        axes[1].set_xlabel("Model")

        st.pyplot(fig)