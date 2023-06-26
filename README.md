# Machine Learning Pipeline for Credit Risk Analysis with the German Credit Data

This repository contains the code and documentation for the CS 412/512 Term Project on Credit Risk Analysis using Machine Learning Algorithms. The project focuses on the application of various classification algorithms to predict credit risk using the German Credit (Statlog) Dataset.
This project was selected as the one that achieved the best accuracy score of mentioned classification problem within the scope of the course.
## Abstract

The aim of this project is to develop a machine learning pipeline for credit risk analysis. We have experimented with various classification algorithms, including Logistic Regression, Decision Trees, Multilayer Perceptron, K-Nearest Neighbors, Support Vector Machines, and Extreme Gradient Boosting. After hyperparameter tuning and model evaluation, we found that Extreme Gradient Boosting (XGB) performed the best, achieving a validation accuracy of 74.33% and a test accuracy of 79%. Additionally, we explored the performance of classifier combinations using Mixture of Experts and Voting algorithms.

## Introduction

The CS412 Machine Learning Term Project focuses on building a practical machine learning pipeline. The project addresses the problem of credit risk assessment using the German Credit (Statlog) Dataset as a benchmark. The dataset consists of 1000 rows and 10 columns, with the last column representing the class label (Good or Bad) and the remaining columns representing features such as age, sex, job, housing, and credit amount.

The goal is to predict whether a loan applicant will have good or bad credit based on the available features. Accurate credit risk assessment can benefit the finance industry by optimizing workforce and reducing expenses.

## Dataset

The German Credit (Statlog) Dataset is a benchmark dataset for credit risk assessment. The dataset is provided in a CSV file format named "german_credit_data.csv" and consists of 1000 rows and 10 columns. The last column represents the class label (Good or Bad), and the remaining columns represent the features of the dataset, such as age, sex, credit history, and job stability.

The dataset includes 700 instances labeled as good credit and 300 instances labeled as bad credit. Prior to applying machine learning algorithms, the dataset undergoes preprocessing steps, including one hot encoding, ordinal encoding, handling missing values, and scaling using MinMaxScaler from the scikit library. After preprocessing, the shape of the feature data is (1000,17), and the label data is (1000,1).

## Methodology

The project implements various classification algorithms for credit risk analysis. The chosen algorithms are Logistic Regression, Decision Trees, Multilayer Perceptron, K-Nearest Neighbors, Support Vector Machines, and Extreme Gradient Boosting. Each algorithm has its strengths and weaknesses, making it important to select the most suitable one for the problem.

Hyperparameter tuning is performed for each algorithm to optimize their performance. Different hyperparameters are tested to achieve the best results. The hyperparameters include regularization parameter, max iterations, penalty function, solver method (for Logistic Regression), criterion, and maximum depth (for Decision Trees), activation function, alpha, hidden layer sizes, max iterations, and solver method (for Multilayer Perceptron), the number of neighbors (for K-Nearest Neighbors), C, gamma, and kernel (for Support Vector Machines), and learning rate, max depth, and number of estimators (for Extreme Gradient Boosting).

## Experiments

The project conducts experiments to evaluate the performance of the implemented algorithms. The results of hyperparameter tuning and model validation are analyzed. Learning curves, accuracy and loss curves, and summary tables are provided.
