import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the training data
data = pd.read_csv("training_data.csv")
texts = data["text"]
labels = data["sentiment"]

# Create and train the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42, stratify=labels
)

model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model on the held-out test split
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy on test set: {accuracy:.0%}")
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Interactive loop
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter a sentence to analyze: ")
    if user_input.lower() == "quit":
        break

    # Transform and predict
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]

    print(f"Sentiment: {prediction}")
