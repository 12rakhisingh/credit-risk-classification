# credit-risk-classification
Module 20 Challenge repo - Rakhi

# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
- This analysis sought to develop and assess machine learning models capable of predicting credit risk, distinguishing between high-risk (1) and healthy (0) loans based on key financial indicators. The primary goal is to create a robust predictive tool for financial institutions to evaluate loan risk and inform informed lending decisions.
  
* Explain what financial information the data was on, and what you needed to predict.
- Financial information from loan applicants is captured in the dataset, including income, loan amount, loan history, and associated risk factors. the goal is to develop a predictive model that can classify loan applications as either low-risk or high-risk based on the provided financial information.
  
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
- Our primary target variable is credit risk, with 0 indicating low risk (healthy loan) and 1 indicating high risk (elevated default risk).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithm).
- The machine learning process in this notebook involves importing data into a DataFrame, separating it into features (X) and target (y), splitting the data into training and testing segments using Scikit-learn's train_test_split, creating and fitting a model with the training data, and finally making predictions and evaluating performance through a confusion matrix and classification report.
This implementation utilizes Logistic Regression from the scikit-learn library as the predictive model, while data evaluation relies on scikit-learn's built-in functions for confusion matrix and classification report.


## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:  

- Accuracy:  
-- Overall accuracy: 0.99 
- Class-wise Performance:  
- Class 0 (Healthy Loans):  
-- Precision: 1.00  
-- Recall: 0.99  
 
- Class 1 (High-Risk Loans):  
-- Precision: 0.84  
-- Recall: 0.94  




## Summary

Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

- The Logistic Regression model demonstrates strong performance in predicting loan credit risk.

Key Findings:

Overall accuracy: 0.99  
Near-perfect precision, recall, and F1-score for healthy loans (Class 0)  
Solid performance for high-risk loans (Class 1), with precision: 0.84, recall: 0.94, and F1-score: 0.89  
Macro and weighted averages show balanced performance across both classes.  
Model Recommendation: Logistic Regression appears to be the most suitable model for this task due to:  

High overall accuracy (0.99)  
Excellent precision and recall for both healthy and high-risk loans  
Balanced results across both classes, as shown by macro and weighted averages.  

Problem-Specific Considerations:

High-risk loans (Class 1): Despite lower precision (0.84), the high recall (0.94) indicates that most high-risk loans are correctly identified, with few false negatives.  
Healthy loans (Class 0): Near-perfect precision and recall ensure that healthy loans are rarely misclassified as high-risk.  

Dependence on Priorities:
If reducing false positives (healthy loans wrongly flagged as high-risk) is a priority, the model’s high precision for Class 0 is advantageous.  
If reducing false negatives (high-risk loans misclassified as healthy) is more critical, the model’s high recall for Class 1 is beneficial.  
Conclusion: Logistic Regression is highly recommended for predicting loan credit risk, offering strong and balanced performance. However, model refinements or hyperparameter tuning can be considered based on specific business priorities, such as minimizing false positives or false negatives.  


If you do not recommend any of the models, please justify your reasoning.
