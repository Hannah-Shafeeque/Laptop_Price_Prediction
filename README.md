# 💻 Laptop Price Prediction Using Machine Learning

This project aims to predict laptop prices based on specifications like brand, processor, RAM, storage, and other features using various machine learning regression models. It includes a user-friendly web interface built with Streamlit.

---

## 📌 Project Objectives

- Predict laptop prices accurately using regression models.
- Analyze and visualize data relationships between features and price.
- Compare multiple machine learning algorithms.
- Build a user interface for real-time price prediction.

---

## 📊 Dataset

The dataset contains various features of laptops such as:

- Brand
- Processor
- RAM
- Storage
- Operating System
- Screen Size
- Graphics
- Weight  

---The dataset used in this project was sourced from Kaggle:
🔗 https://drive.google.com/file/d/1uVLUxVS5RQXdeECcTbo0Xl2-x8eZ3HSt/view?usp=sharing


## 🔧 Technologies & Tools Used

- **Programming Language**: Python
- **Data Manipulation**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: scikit-learn
    - Algorithms: KNN, Decision Tree, Linear Regression, SVM, Random Forest, AdaBoost
    - Model Evaluation: R² Score, Mean Absolute Error
    - Preprocessing: LabelEncoder, OneHotEncoder, MinMaxScaler
    - Feature Engineering: PolynomialFeatures
    - Hyperparameter Tuning: GridSearchCV
- **Web Interface**: Streamlit
- **Environment**: Google colab , VS Code

---

## 🚀 Workflow

1. **Data Preprocessing**  
   - Handle missing values, encode categorical features, and scale data.
   
2. **Exploratory Data Analysis (EDA)**  
   - Visualize relationships and check for outliers and correlations.

3. **Model Training & Evaluation**  
   - Train multiple regression models and evaluate using R² score.
   - Apply hyperparameter tuning with GridSearchCV.

4. **Model Comparison**  
   - Compare model performances using visual plots.

5. **Web App Deployment**  
   - Deploy the best-performing model (Random Forest Regressor) using Streamlit.

---

## 🎯 Results

| Model             | R² Score |
|------------------|----------|
| KNN              | 0.75     |
| Decision Tree    | 0.97     |
| Linear Regression| 0.96     |
| SVM              | 0.94     |
| Random Forest    | 0.98 ✅ (Best) |
| AdaBoost         | 0.90     |

---

## 🧠 What I Learned

- How to preprocess real-world data and engineer useful features.
- Differences between regression models and their strengths.
- Importance of model evaluation and hyperparameter tuning.
- How to build a real-time ML-powered web interface using Streamlit.


