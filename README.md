# Zomato Restaurant Rating Prediction

live demo:https://zomato-restaurant-rating-prediction.onrender.com/

## Project Summary

This project focuses on predicting restaurant ratings on Zomato by leveraging various data sources, including restaurant metadata, user reviews, and incorporating sentiment analysis of review text. The project follows a comprehensive machine learning pipeline, from data loading and cleaning to advanced feature engineering, training and comparing multiple predictive models, and evaluating their performance. A key objective is to identify the most influential factors affecting restaurant ratings, including the role of sentiment derived from reviews. Additionally, the project explores the relationship between cuisine, cost, and sentiment to provide deeper business insights.

## Problem Statement

The goal is to build a robust predictive model that can accurately estimate a restaurant's rating on the Zomato platform. This involves:
1.  Loading and cleaning disparate datasets containing restaurant information and user reviews.
2.  Performing exploratory data analysis to understand the data distribution and relationships.
3.  Engineering relevant features, including sentiment from review text and structured metadata.
4.  Building and comparing various machine learning regression models.
5.  Optimizing the best-performing model's hyperparameters.
6.  Identifying the key features that drive the model's predictions.
7.  Providing insights into the relationship between sentiment, cost, and cuisines.

## Dataset Sources

1.  `Zomato Restaurant names and Metadata.csv`
2.  `Zomato Restaurant reviews.csv`

These datasets should be placed in the `/content/` directory or the file paths in the notebook code should be updated accordingly.

## Methodology

The project follows these main steps:

1.  **Data Loading and Initial Inspection:** Loading the metadata and reviews datasets and examining their structure and initial quality.
2.  **Data Wrangling & Cleaning:** Handling missing values, removing duplicates, and correcting data types in both datasets.
3.  **Merge DataFrames:** Combining the cleaned metadata and reviews data into a single DataFrame based on restaurant names.
4.  **Exploratory Data Analysis (EDA) & Visualization:** Analyzing data distributions and relationships through visualizations to gain insights into factors influencing ratings.
5.  **Hypothesis Testing:** Conducting statistical tests (ANOVA, t-tests) to formally validate observed relationships, such as the difference in ratings across cost brackets and between specific cuisines.
6.  **Feature Engineering & Data Pre-processing:** Creating new features from existing data, including sentiment polarity from review text, numerical features from metadata, temporal features from review timestamps, and one-hot encoding for categorical variables. Scaling numerical features using `StandardScaler`.
7.  **ML Model Building and Comparison:** Training and evaluating multiple regression models (Linear Regression, Decision Tree, Random Forest, XGBoost) to predict restaurant ratings.
8.  **Hyperparameter Optimization:** Using `RandomizedSearchCV` to fine-tune the hyperparameters of the best-performing model (XGBoost).
9.  **Model Explainability and Future Work:** Analyzing feature importance to understand the model's predictions and outlining potential directions for future research and improvements.
10. **Model Saving:** Saving the best-performing model using `joblib` for potential deployment.

## Key Findings and Results

*   The distribution of restaurant ratings is skewed towards higher values.
*   There is a statistically significant difference in average ratings across different cost brackets and between 'North Indian' and 'Chinese' cuisines.
*   Feature engineering, especially the inclusion of sentiment polarity from reviews, significantly improved model performance.
*   The **Tuned XGBoost Regressor** was the best-performing model, achieving an R-squared of approximately **0.569** on the test set.
*   Feature importance analysis revealed that **Sentiment Polarity**, **Reviewer Metrics** (review and follower counts), and **Temporal Features** (day of the week, month) are the most influential factors in predicting restaurant ratings. Cost also plays a significant role.

## How to Run the Code

1.  Clone this repository to your local machine or open it in Google Colab.
2.  Ensure the dataset files (`Zomato Restaurant names and Metadata.csv` and `Zomato Restaurant reviews.csv`) are placed in the correct directory as specified in the notebook (e.g., `/content/`).
3.  Run the cells in the provided Jupyter Notebook sequentially. The notebook contains all the necessary code for data loading, cleaning, feature engineering, model training, evaluation, optimization, and explainability.
4.  Ensure you have the required libraries installed (e.g., pandas, numpy, matplotlib, seaborn, scipy, sklearn, xgboost, joblib, textblob). You can install them using pip if needed:
