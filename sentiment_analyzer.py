import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


texts = [
    "I love this product! It's amazing.",
    "This is the worst experience ever.",
    "I am so happy with my purchase.",
    "I am very disappointed with the service.",
    "This is a great product.",
    "I am not satisfied with the product.",
    "I am very happy with the service.",
    "I am very disappointed with the product.",
    "This is an amazing product.",
    "I am not satisfied with the service."
]  

labels = [
    "positive",
    "negative",
    "positive", 
    "negative",
    "positive",
    "negative",
    "positive",
    "negative",
    "positive",
    "negative"
]

# Create and train the model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
print("Type 'quit' to exit\n")

while True:
  user_input = input("Enter a sentence to analyze: ")
  if user_input.lower() == "quit":
      break





# transform and predict
  input_vector = vectorizer.transform([user_input])
  prediction = model.predict(input_vector)[0]

  print(f"Sentiment: {prediction}")

