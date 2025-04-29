import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("student_performance_1200.csv")
target_column = "Final Score"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set Streamlit page config
st.set_page_config(page_title="Student Score Predictor", layout="wide")

# Sidebar
theme = st.sidebar.selectbox("🌗 Select Theme", ["Light", "Dark"])
model_choice = st.sidebar.selectbox("🤖 Choose Model", ["Random Forest", "Linear Regression", "Decision Tree"])

# Theme styles
dark_theme_css = """
    <style>
    .stApp { background-color: #1E1E1E; color: white; }
    .main { background-color: rgba(0, 0, 0, 0.6); padding: 20px; border-radius: 10px; }
    h1, .streamlit-expanderHeader { color: #FFD700; }
    </style>
"""
light_theme_css = """
    <style>
    .stApp { background-color: #FFFFFF; color: black; }
    .main { background-color: rgba(255, 255, 255, 0.85); padding: 20px; border-radius: 10px; }
    h1, .streamlit-expanderHeader { color: #1E1E1E; }
    </style>
"""
st.markdown(dark_theme_css if theme == "Dark" else light_theme_css, unsafe_allow_html=True)

# Model loader
def get_model(name):
    if name == "Random Forest":
        model = RandomForestRegressor()
    elif name == "Linear Regression":
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

model = get_model(model_choice)
y_pred_test = model.predict(X_test)

# Metrics
st.sidebar.subheader("📈 Evaluation Metrics")
st.sidebar.metric("R² Score", f"{r2_score(y_test, y_pred_test):.2f}")
st.sidebar.metric("MAE", f"{mean_absolute_error(y_test, y_pred_test):.2f}")
st.sidebar.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# Header
st.title("🎓 Student Score Predictor")
st.write("Predict your final exam score based on lifestyle and academic habits.")

# Input form (dynamically generated from columns)
input_values = []
col1, col2 = st.columns(2)
features = X.columns.tolist()

# Mapping input widgets to expected columns
with col1:
    study_hours = st.slider("📘 Study Hours per Day", 0.0, 6.0, 3.0)
    attendance = st.slider("🏫 Attendance (%)", 50, 100, 80)
    sleep_hours = st.slider("😴 Sleep Hours", 4.0, 10.0, 7.0)

with col2:
    extracurricular = st.selectbox("🎭 Extracurricular Participation", [0, 1])
    prev_gpa = st.slider("📚 Previous GPA", 0.0, 10.0, 6.0)
    internet_usage = st.slider("🌐 Internet Usage (hrs/day)", 0.0, 5.0, 2.0)

# Collect input into correct format
input_data = pd.DataFrame([[study_hours, attendance, sleep_hours, extracurricular, prev_gpa, internet_usage]],
                          columns=features)

# Predict button
if st.button("🔮 Predict Score"):
    try:
        predicted_score = model.predict(input_data)[0]
        st.subheader("🎯 Predicted Final Exam Score")
        st.metric("Predicted Score", f"{predicted_score:.2f}")

        with st.expander("🔍 Explanation"):
            st.write(f"The **{model_choice}** model predicts your score based on the input features you provided.")

        # Study Hours vs Final Score
        st.subheader("📊 Study Hours vs Final Score")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(df['Study Hours'], df['Final Score'], alpha=0.6)
        ax1.set_xlabel("Study Hours")
        ax1.set_ylabel("Final Score")
        ax1.set_title("Study Hours vs Final Score")
        st.pyplot(fig1)

        # Correlation Heatmap
        st.subheader("🔍 Feature Correlation Heatmap")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
        st.pyplot(fig2)

        # Score Distribution
        st.subheader("📈 Predicted Score Distribution")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.histplot(model.predict(X), kde=True, bins=30, color="green", ax=ax3)
        ax3.set_xlabel("Predicted Scores")
        ax3.set_title("Distribution of All Predicted Scores")
        st.pyplot(fig3)

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("""
    <footer style="text-align: center; color: gray;">
        <p>🧠 Built with Streamlit | 🔢 Multiple Models Supported | 🎨 Theme Toggle</p>
    </footer>
""", unsafe_allow_html=True)
