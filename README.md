# Sentiment Analyzer

A command-line tool built in Python to classify a piece of text as having 'positive', 'negative', or 'neutral' sentiment. This project uses the `scikit-learn` library to train a simple machine learning model for natural language processing (NLP).

## Features

-   Trains on a labeled dataset (`training_data.csv`) of example sentences.
-   Reports the trained model's accuracy and a per-class precision/recall breakdown on a held-out test split.
-   Interactively analyzes any sentence you type in and predicts its sentiment.

## How It Works

The script performs the following steps:
1.  **Loads the training dataset** (`training_data.csv`) using `pandas` — a set of text samples labeled 'positive', 'negative', or 'neutral'.
2.  **Vectorizes the text data:** It converts the text into numerical features using `CountVectorizer`, which creates a matrix of token counts.
3.  **Trains a Classifier:** A `Multinomial Naive Bayes` classifier, which is well-suited for text classification tasks, is trained on a split of the vectorized data.
4.  **Evaluates the model:** The remaining held-out data is used to report accuracy and a classification report (precision/recall/F1 per class).
5.  **Makes Predictions:** For each sentence you type in, the script transforms it using the same vectorizer and feeds it to the trained model to predict the sentiment.

**Note on accuracy:** `training_data.csv` only has 19 example sentences, so this is a small, educational demonstration of the pipeline rather than a production-grade classifier — accuracy on the test split will vary a lot run to run and won't be high. Predictions are most reliable for sentences that share vocabulary with the training examples. Adding more labeled examples to `training_data.csv` is the easiest way to improve it.

## Installation

To get this project running on your local machine, follow these steps.

**Prerequisites:**
- Python 3.6+
- Git

**Setup:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MikeM365/sentiment-analyzer.-
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd sentiment-analyzer.-
    ```

3.  **Create and activate a virtual environment:**
    - On **Windows**:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - On **macOS / Linux**:
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script from your terminal. It will train, print an accuracy report, then prompt you for sentences to analyze. Type `quit` to exit.

```bash
python sentiment_analyzer.py
```

**Example session:**
```
Model accuracy on test set: 75%

Classification report:
              precision    recall  f1-score   support
    negative       1.00      1.00      1.00         2
     neutral       1.00      1.00      1.00         1
    positive       0.00      0.00      0.00         1
...

Type 'quit' to exit

Enter a sentence to analyze: This is amazing and wonderful!
Sentiment: positive
Enter a sentence to analyze: quit
```

(Exact accuracy numbers will vary between runs since the test split is small.)

## Dependencies

-   Python 3
-   pandas
-   scikit-learn
