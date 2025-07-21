# Sentiment Analyzer

A command-line tool built in Python to classify a given piece of text as having 'Positive' or 'Negative' sentiment. This project uses the `scikit-learn` library to train a simple machine learning model for natural language processing (NLP).

## Features

-   Analyzes input text provided via the command line.
-   Outputs a clear prediction: 'Positive' or 'Negative'.
-   Uses a simple and effective machine learning pipeline.

## How It Works

The script performs the following steps:
1.  **Loads a pre-defined dataset** of text samples labeled with their sentiment.
2.  **Vectorizes the text data:** It converts the text into numerical features using `CountVectorizer`, which creates a matrix of token counts.
3.  **Trains a Classifier:** A `Multinomial Naive Bayes` classifier, which is well-suited for text classification tasks, is trained on the vectorized data.
4.  **Makes a Prediction:** When you provide a new sentence, the script transforms it using the same vectorizer and feeds it to the trained model to predict the sentiment.

## Installation

To get this project running on your local machine, follow these steps.

**Prerequisites:**
- Python 3.6+
- Git

**Setup:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
    ```
    *(Note: Replace `YOUR_USERNAME` with your actual GitHub username.)*

2.  **Navigate to the project directory:**
    ```bash
    cd sentiment-analyzer
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

To run the analyzer, execute the main script from your terminal, passing the text you want to analyze as an argument in quotes.

**Syntax:**
```bash
python sentiment_analyzer.py "Your text to analyze here"

Example:

python sentiment_analyzer.py "This movie was fantastic and I really enjoyed the acting."

Expected Output:

Predicted Sentiment: Positive

Dependencies

Python 3
scikit-learn
