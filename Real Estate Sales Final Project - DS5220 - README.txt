Real Estate Sales 2001-2020

Class: Supervised Machine Learning and Learning Theory

Team Members: Ikonkar Kaur Khalsa and Matthew Gregorio

Project Description: This report presents a comprehensive analysis aimed at developing accurate and interpretable machine learning models for predicting house sale amounts and classifying property affordability. Utilizing an extensive dataset from Kaggle, encompassing over a million real estate transactions across two decades, the study evaluates both regression and classification models to identify the most effective predictive approach.

The methodology involves preprocessing, exploratory data analysis, and training various algorithms, including:

-Regression Models: Linear, Lasso, and Ridge

-Classification Models: Logistic Regression, Decision Trees, Gaussian Naive Bayes, and Neural Networks

The comparative analysis reveals that the Decision Tree model, specifically the second variant, outperformed others in terms of model evaluation scores and interpretability, providing significant insights into the impact of features such as Longitude on sale amounts. Regression models demonstrated inferior performance, indicating that classification approaches were more suitable for this dataset.

Installation: This project is developed using Google Colab and can also be run on other platforms such as Visual Studio Code.

The following libraries and frameworks need to be imported:

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear.model import Lasso

from sklearn.linear.model import LogisticRegression

import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.tree import DecisionTreeClassifier, plot_tree

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam

from sklearn.naive_bayes import GaussianNB

Usage:

To use this project:

-Clone the repository.

-Open the project in Google Colab or your preferred Python environment.

-Ensure all the required libraries are installed.

-Run the cells in sequence to preprocess the data, train the models, and evaluate their performance.

Features:

-Preprocessing and exploratory data analysis of real estate transaction data.

-Implementation of various machine learning algorithms for both regression and classification.

-Comprehensive model evaluation and comparison.

-Visualization of results and model interpretability.

Technologies Used: Python Pandas Scikit-Learn Matplotlib Seaborn TensorFlow Keras Contributing

Contributions are welcome! If you have suggestions for new machine learning algorithms that could improve this project or any other enhancements, please fork the repository and create a pull request.

Contact Information:

Ikonkar Kaur Khalsa: khalsa.i@northeastern.edu

Matthew Gregorio: gregorio.m@northeastern.edu

Acknowledgments: We would like to thank our professor and peers for their support and guidance throughout this project.