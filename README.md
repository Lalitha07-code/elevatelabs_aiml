# 📁 Titanic Survival Prediction Project

---

## 🎯 Objective:
To understand and prepare the Titanic dataset for building a predictive machine learning model. This includes data cleaning, exploratory data analysis (EDA), handling missing values and outliers, and drawing inferences from visualizations.

---

## ✅ Task 1: Data Cleaning & Preprocessing

### 🔍 Goals:
- Handle missing values appropriately.
- Treat outliers to prevent distortion in modeling.
- Perform initial data checks.

### 📌 Key Steps:
1. **Loaded the Titanic dataset.**
2. **Handled Missing Values:**
   - `Cabin` filled with `'Unknown'`.
   - `Age` and `Fare` filled with median values.
3. **Outlier Detection & Removal:**
   - Detected using boxplots for `Age` and `Fare`.
   - Removed entries where `Fare > 300` and `Age > 80`.

### 🧹 Outcome:
- Dataset cleaned and ready for exploratory analysis.
- Missing values and outliers no longer distort the feature distributions.

---

## ✅ Task 2: Exploratory Data Analysis (EDA)

### 🔍 Goals:
- Analyze data distributions and relationships between features.
- Use visualizations to infer patterns and trends.

### 📊 Key Steps:
1. **Generated Summary Statistics:**
   - Mean, median, standard deviation helped understand spread.

2. **Plotted Visuals:**
   - **Histograms** for `Age` and `Fare`.
   - **Boxplots** to detect outliers.
   - **Bar plots** comparing `Survived` with `Sex` and `Pclass`.
   - **Correlation Heatmap** to find relationships between numeric features.
   - **Pairplot** visualizing feature relationships by `Survived`.

3. **Handled Categorical Encoding:**
   - Used one-hot encoding for categorical features (`Sex`, `Embarked`, etc.).

---

## 🧠 Inference Summary (From Visuals):

- **Survival Rate by Sex:** Females had a significantly higher survival rate than males.
- **Survival Rate by Class:** 1st class passengers had better survival odds.
- **Fare:** Right-skewed; higher fares generally linked to survival.
- **Age:** Most passengers were 20–40 years old; survival scattered across ages.
- **Correlation:** Strong negative correlation between `Pclass` and `Fare`; moderate positive correlation between `SibSp` and `Parch`.

---
# 🏠task 3 House Price Prediction (Linear Regression)

A simple ML project that predicts house prices using Linear Regression on `Housing.csv`.

## 🔧 Steps
1. Load & preprocess data (encoding categoricals)
2. Train-test split
3. Train Linear Regression model
4. Evaluate using MAE, MSE, R²
5. Plot actual vs predicted, view coefficients

## 📊 Sample Results
- MAE: ~₹9.7 L
- R² Score: ~0.65

## 📦 Libraries
pandas, numpy, sklearn, matplotlib, seaborn

# task 4 Logistic Regression - Breast Cancer Classification

This project demonstrates how to apply **Logistic Regression** for binary classification on the **Breast Cancer Wisconsin dataset**.

## Project Steps
1. **Dataset Selection**
   - Uses `data.csv` (Breast Cancer Wisconsin dataset).
2. **Data Preprocessing**
   - Dropped unnecessary columns: `id`, `Unnamed: 32`
   - Encoded target variable: `M` → 1 (Malignant), `B` → 0 (Benign)
3. **Train/Test Split & Standardization**
   - Used 80% training, 20% testing
   - Standardized features using `StandardScaler`
4. **Model Training**
   - Trained a Logistic Regression model using scikit-learn
5. **Model Evaluation**
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC score
   - ROC Curve plot
6. **Threshold Tuning**
   - Example with threshold = 0.3 to show effect on precision and recall
7. **Sigmoid Function Explanation**
   - Plotted sigmoid curve
   - Explained its role in converting model outputs to probabilities

     # Heart Disease Prediction - Decision Tree & Random Forest

## 📌Task 5 Project Overview
This project uses the **Heart Disease Dataset** to train and evaluate two machine learning models:
- **Decision Tree Classifier**
- **Random Forest Classifier**

We:
1. Train and visualize a Decision Tree.
2. Analyze overfitting by controlling tree depth.
3. Train a Random Forest and compare accuracy.
4. Interpret feature importances.
5. Evaluate both models using cross-validation.

---

## 📂 Dataset
The dataset used is `heart.csv`, which contains various medical attributes such as:
- Age, sex, chest pain type, blood pressure, cholesterol, etc.
- Target: `1` → Heart disease, `0` → No heart disease.

---

# TASK 6 KNN Classification on Iris Dataset

## Steps Performed
1. **Dataset**  
   - Used `Iris.csv` dataset.  
   - Dropped ID column.  
   - Encoded species labels into numbers.  
   - Normalized features using `StandardScaler`.  

2. **Model**  
   - Applied `KNeighborsClassifier` from `sklearn`.  
   - Tested different `K` values to find the best accuracy.  

3. **Evaluation**  
   - Computed accuracy score.  
   - Created confusion matrix.  

4. **Visualization**  
   - Plotted decision boundaries using first two normalized features.  
   - Colors: Red (Setosa), Green (Versicolor), Blue (Virginica).  

## Output
- Best K value  
- Accuracy score  
- Confusion matrix plot  
- Decision boundary plot









