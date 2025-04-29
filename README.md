# 🎓 Student Score Predictor

A user-friendly web app built using **Streamlit** that predicts a student's final exam score based on lifestyle and academic habits such as study hours, sleep duration, attendance, GPA, and more. Multiple machine learning models are supported, and users can switch between Light and Dark themes.

---

## 🚀 Features

- 📊 Predict final exam scores using:
  - Random Forest Regressor
  - Linear Regression
  - Decision Tree Regressor
- 🎨 Toggle between Light and Dark themes
- 📈 Real-time evaluation metrics (R², MAE, RMSE)
- 🔍 Visualizations:
  - Study Hours vs Final Score
  - Correlation Heatmap
  - Predicted Score Distribution
- 🤖 Intuitive and interactive UI using Streamlit

---

## 📁 Dataset

The app uses a CSV file named `student_performance_1200.csv` containing features like:

- Study Hours
- Attendance (%)
- Sleep Hours
- Extracurricular Participation (0/1)
- Previous GPA
- Internet Usage (hrs/day)
- Final Score (target variable)

> Make sure this CSV is in the root directory of your repo for successful deployment on Streamlit Cloud.

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

---

## 🛠 How to Run Locally

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/student-score-predictor.git
    cd student-score-predictor
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub and deploy this repo.
4. Done! 🚀

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
