# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- Main function to run the entire process ---
def run_spam_detector():
    # 2. Load and Prepare the Data
    # The dataset is in the 'data' subfolder
    file_path = 'data/spam.csv'
    df = pd.read_csv(file_path, encoding='latin-1')

    # Keep only necessary columns and rename them for clarity
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # Convert labels to numerical format: 'ham' -> 0, 'spam' -> 1
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    print("--- Data Loaded and Prepared ---")
    print(df.head())

    # 3. Split Data into Training and Testing sets
    X = df['message']  # The feature is the message text
    y = df['label_num'] # The target is the spam/ham label

    # Split data into 80% for training and 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Feature Extraction (Convert text to numbers)
    # Use TF-IDF to find the importance of words
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    
    # Fit and transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    # Only transform the test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\n--- Text Converted to Numerical Features ---")

    # 5. Train the Machine Learning Model
    # Naive Bayes is a great choice for text classification
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    print("--- Model Trained Successfully ---")

    # 6. Evaluate the Model
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Model Evaluation ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham (Not Spam)', 'Spam']))

    # 7. Test with a New Message
    print("\n--- Testing with New Messages ---")
    
    def predict_message(message):
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]
        prediction_label = "SPAM" if prediction == 1 else "NOT SPAM (Ham)"
        print(f"Message: '{message}'")
        print(f"Prediction: ** {prediction_label} **\n")

    # Test with a spammy message
    predict_message("Congratulations you won a free vacation! Click here to claim your prize now!")
    # Test with a normal message
    predict_message("Hey can you send me the report by 5pm? Thanks.")

# --- This part makes the script runnable from the command line ---
if __name__ == '__main__':
    run_spam_detector()