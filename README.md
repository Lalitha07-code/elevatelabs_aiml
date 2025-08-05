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

## 🧾 Conclusion:
Through systematic data cleaning and insightful EDA:
- We understood the data better and prepared it for modeling.
- Identified key features impacting survival.
- This foundation will support better feature engineering and model accuracy in the next task.

