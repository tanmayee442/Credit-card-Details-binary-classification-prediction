# Credit Card Default Prediction - Binary Classification

This project aims to predict whether a customer is likely to **default on their credit card payment** using a machine learning model. It is a binary classification task, where the target label is:

**`0`** → No Default

**`1`** → Default

The project uses a **Random Forest Classifier** trained on cleaned and encoded credit card customer data.

##  Dataset Files

**`Credit_card.csv`**: Contains credit card customer features.

**`Credit_card_label.csv`**: Contains the corresponding labels for default (0 or 1).

##  Model Overview

**Algorithm Used:** Random Forest Classifier
 
**Preprocessing:**
Missing value handling (mode for categorical, median for numerical)

Label Encoding for categorical features

Merging features and labels via `Ind_ID`
  
**Train-Test Split:** 80% training / 20% testing

**Evaluation Metrics:**

Accuracy

Classification Report

Confusion Matrix

##  Visualizations

**The project includes the following plots:**

Class Distribution (for imbalance detection)

Confusion Matrix

Missing Values Heatmap (before encoding)

Correlation Heatmap (after encoding)

Feature Importance Graph (from Random Forest)

##  How to Run

### 1. Clone the repository

git clone https://github.com/tanmayee442/Credit-card-Details-binary-classification-prediction.git

cd Credit-card-Details-binary-classification-prediction

### 2. Install the Required Libraries

Make sure Python is installed, then run:

pip install pandas scikit-learn matplotlib seaborn

### 3. Run the Script

python Credit_Debit.py

**This script will:**

Load the datasets

Preprocess the data

Train a Random Forest classifier

Print evaluation results

Show multiple plots for insight and interpretation

### Libraries Used

**pandas** – for data manipulation

**scikit-learn** – for machine learning and preprocessing

**matplotlib** – for plotting

**seaborn** – for styled visualizations

 
