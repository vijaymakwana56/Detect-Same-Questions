# 🚀 Similar Question Detection App (Streamlit)

This is a simple web app built using **Streamlit** that detects whether two input questions are semantically similar or not using a machine learning model trained on question pairs.

---

## 🖥️ Live App

👉 [Try the app here](https://detect-same-questions-dsjcqsenjcwjtaqqfx5xtx.streamlit.app/)

---

## 📦 Features

- Input two questions.
- Preprocess and vectorize using a trained CountVectorizer or TF-IDF model.
- Predict similarity using a pre-trained ML model (e.g., XGBoost / SVM / Logistic Regression).
- Clean and responsive UI powered by Streamlit.

---

## 📁 Project Structure

├── app.py # Main Streamlit app
├── model.pkl # Trained ML model
├── vectorizer.pkl # CountVectorizer or TF-IDF model
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 🧠 Model Info

- Trained using: `scikit-learn` or `xgboost`
- Input features: Bag-of-Words / TF-IDF
- Preprocessing: HTML cleaning, lowercasing, token filtering, etc.

---

## ⚙️ How to Run Locally

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-username/similar-question-app.git
   cd similar-question-app

Install dependencies:
pip install -r requirements.txt

Run the app:
streamlit run app.py

