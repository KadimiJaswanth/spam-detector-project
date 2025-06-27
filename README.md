# Spam Email / SMS Detector

## Project Description
This is a simple machine learning project that classifies messages as either "Spam" or "Ham" (not spam). The model is trained on the popular SMS Spam Collection Dataset from UCI. It uses a `TfidfVectorizer` to convert text data into numerical features and a `Multinomial Naive Bayes` classifier to make predictions.

## Dataset
The dataset used is the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle.

## How to Run
1.  Clone this repository.
2.  Create a virtual environment: `python -m venv venv`
3.  Activate it: `source venv/bin/activate` (on Mac/Linux) or `venv\Scripts\activate` (on Windows).
4.  Install the required libraries: `pip install -r requirements.txt`
5.  Run the main script: `python spam_detector.py`

## Results
The model achieved an accuracy of approximately **98.3%** on the test set.

**Classification Report:**
```
                precision    recall  f1-score   support

Ham (Not Spam)       0.98      1.00      0.99       966
          Spam       0.99      0.89      0.94       149
```
