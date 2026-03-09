# Sentiment-Analysis-Tool


A machine learning program that detects if text is positive or negative.

## Features
- Uses 3 different models: Naive Bayes, Logistic Regression, Random Forest
- Compares model performance
- Interactive command line interface
- Creates visualizations of data and results
- Saves trained model for reuse

## How to Run
1. Install required packages:
   
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud joblib requests


3. Download NLTK data:

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"



3. Run the program:

python sentiment.py



## Commands
- `analyze: your text here` - Analyze specific text
- `test` - Run sample test cases
- `stats` - View model performance
- `exit` - Quit

## Example Usage
analyze: I love this product!
Result: POSITIVE 😊 (confidence: 0.99)




Text: I love this product!
Sentiment: POSITIVE 😊
Confidence: 0.99



## Output Files
- `sentiment_model.pkl` - Saved trained model
- `data_exploration.png` - Data visualization
- `model_comparison.png` - Model performance comparison

## Results
All three models achieve 95-100% accuracy on the test data.
