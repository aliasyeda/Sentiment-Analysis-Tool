# sent.py
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import requests
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                           confusion_matrix)
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import os

class ProfessionalSentimentAnalyzer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        self.results = {}
        
    def download_dataset(self):
        """Try to download a dataset from working URLs"""
        print("📥 Attempting to download dataset...")
        
        # Working dataset URLs
        urls = [
            "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv",
            "https://raw.githubusercontent.com/skathirmani/datasets/master/twitter_sentiment.csv",
            "https://raw.githubusercontent.com/amephraim/nlp/master/text_classification/data/reviews.csv"
        ]
        
        for url in urls:
            try:
                print(f"Trying: {url}")
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    # Save the file
                    with open('dataset.csv', 'wb') as f:
                        f.write(response.content)
                    print("✅ Dataset downloaded successfully!")
                    
                    # Try to read it
                    df = pd.read_csv('dataset.csv')
                    
                    # Check columns and rename if needed
                    if 'text' not in df.columns and 'sentiment' not in df.columns:
                        # Try to identify text and sentiment columns
                        for col in df.columns:
                            if df[col].dtype == 'object' and 'text' not in df.columns:
                                df = df.rename(columns={col: 'text'})
                            elif df[col].dtype in ['int64', 'float64'] and 'sentiment' not in df.columns:
                                df = df.rename(columns={col: 'sentiment'})
                    
                    if 'text' in df.columns and 'sentiment' in df.columns:
                        print(f"✅ Successfully loaded dataset with {len(df)} samples")
                        return df
            except Exception as e:
                print(f"Failed: {str(e)[:50]}...")
                continue
        
        print("⚠️ Using enhanced sample dataset")
        return self.create_enhanced_dataset()
    
    def create_enhanced_dataset(self):
        """Create a more realistic dataset with variation"""
        import random
        
        positive_words = ['amazing', 'great', 'excellent', 'fantastic', 'wonderful', 
                         'love', 'perfect', 'best', 'happy', 'satisfied']
        negative_words = ['terrible', 'worst', 'poor', 'bad', 'awful', 
                         'disappointed', 'horrible', 'waste', 'regret', 'useless']
        
        data = []
        
        # Create 500 positive reviews with variations
        for i in range(500):
            words = random.sample(positive_words, 3)
            templates = [
                f"This product is {words[0]} and {words[1]}!",
                f"Absolutely {words[0]} experience, very {words[1]}.",
                f"{words[0].capitalize()} quality and {words[1]} service!",
                f"I'm so {words[1]} with this purchase, it's {words[0]}!",
                f"Really {words[0]} and {words[1]}. Highly recommended!"
            ]
            text = random.choice(templates)
            # Add some noise (not all positive)
            if random.random() < 0.05:  # 5% chance of negative word
                text += f" But {random.choice(negative_words)} shipping."
            data.append({'text': text, 'sentiment': 1})
        
        # Create 500 negative reviews with variations
        for i in range(500):
            words = random.sample(negative_words, 3)
            templates = [
                f"This product is {words[0]} and {words[1]}!",
                f"Absolutely {words[0]} experience, very {words[1]}.",
                f"{words[0].capitalize()} quality and {words[1]} service!",
                f"I'm so {words[1]} with this purchase, it's {words[0]}!",
                f"Really {words[0]} and {words[1]}. Would not recommend!"
            ]
            text = random.choice(templates)
            # Add some noise (not all negative)
            if random.random() < 0.05:  # 5% chance of positive word
                text += f" But {random.choice(positive_words)} packaging."
            data.append({'text': text, 'sentiment': 0})
        
        df = pd.DataFrame(data)
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        print(f"✅ Created realistic dataset with {len(df)} samples")
        return df
    
    def advanced_preprocessing(self, text):
        """Professional text preprocessing"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove special chars and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def explore_data(self, df):
        """Generate and save data visualizations without displaying"""
        print("\n📊 DATA EXPLORATION")
        print("="*50)
        print(f"Total samples: {len(df)}")
        print(f"\nSentiment distribution:")
        print(df['sentiment'].value_counts())
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Sentiment distribution
        plt.subplot(2, 3, 1)
        df['sentiment'].value_counts().plot(kind='bar', color=['red', 'green'])
        plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
        plt.xticks([0, 1], ['Negative', 'Positive'], rotation=0)
        plt.ylabel('Count')
        
        # 2. Text length distribution
        plt.subplot(2, 3, 2)
        df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
        df[df['sentiment']==1]['text_length'].hist(alpha=0.7, label='Positive', color='green', bins=20)
        df[df['sentiment']==0]['text_length'].hist(alpha=0.7, label='Negative', color='red', bins=20)
        plt.title('Text Length by Sentiment', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 3. Pie chart
        plt.subplot(2, 3, 3)
        sizes = df['sentiment'].value_counts()
        plt.pie(sizes, labels=['Positive', 'Negative'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90)
        plt.title('Sentiment Ratio', fontsize=14, fontweight='bold')
        
        # 4. Create word clouds
        positive_text = ' '.join(df[df['sentiment']==1]['text'].head(100).apply(self.advanced_preprocessing))
        negative_text = ' '.join(df[df['sentiment']==0]['text'].head(100).apply(self.advanced_preprocessing))
        
        # Positive word cloud
        plt.subplot(2, 3, 4)
        if positive_text.strip():
            wordcloud_pos = WordCloud(width=400, height=200, background_color='white', 
                                     colormap='Greens').generate(positive_text)
            plt.imshow(wordcloud_pos, interpolation='bilinear')
            plt.title('Positive Reviews - Word Cloud', fontsize=14, fontweight='bold')
            plt.axis('off')
        
        # Negative word cloud
        plt.subplot(2, 3, 5)
        if negative_text.strip():
            wordcloud_neg = WordCloud(width=400, height=200, background_color='white', 
                                     colormap='Reds').generate(negative_text)
            plt.imshow(wordcloud_neg, interpolation='bilinear')
            plt.title('Negative Reviews - Word Cloud', fontsize=14, fontweight='bold')
            plt.axis('off')
        
        # 5. Sample texts
        plt.subplot(2, 3, 6)
        plt.axis('tight')
        plt.axis('off')
        sample_texts = []
        for i in range(3):
            pos_sample = df[df['sentiment']==1]['text'].iloc[i][:50] + "..."
            neg_sample = df[df['sentiment']==0]['text'].iloc[i][:50] + "..."
            sample_texts.append([f"✓ {pos_sample}", f"✗ {neg_sample}"])
        
        table = plt.table(cellText=sample_texts,
                         colLabels=['Positive Samples', 'Negative Samples'],
                         cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Sample Reviews', fontsize=14, fontweight='bold')
        
        plt.suptitle('Sentiment Analysis Dataset Exploration', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save instead of show
        plt.savefig('data_exploration.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📸 Data exploration saved as 'data_exploration.png'")
        
        return df
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test):
        """Train and compare multiple models"""
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', model)
            ])
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            
            # Cross-validation (using 3 folds for speed)
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3)
            
            # Store results
            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred
            }
            
            print(f"  ✅ Accuracy: {results[name]['accuracy']:.4f}")
            print(f"  ✅ F1-Score: {results[name]['f1']:.4f}")
            print(f"  ✅ CV Score: {results[name]['cv_mean']:.4f} (+/- {results[name]['cv_std']:.4f})")
        
        self.results = results
        return results
    
    def compare_models(self, y_test):
        """Create model comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Accuracy Comparison
        names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in names]
        f1_scores = [self.results[name]['f1'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[0,0].bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue')
        axes[0,0].bar(x + width/2, f1_scores, width, label='F1-Score', color='lightgreen')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('Score')
        axes[0,0].set_title('Model Performance Comparison', fontweight='bold')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(names, rotation=45)
        axes[0,0].legend()
        axes[0,0].set_ylim(0, 1)
        
        # 2. Confusion Matrix for best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        cm = confusion_matrix(y_test, self.results[best_model_name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1])
        axes[0,1].set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # 3. Cross-validation scores
        cv_means = [self.results[name]['cv_mean'] for name in names]
        cv_stds = [self.results[name]['cv_std'] for name in names]
        
        axes[1,0].bar(names, cv_means, yerr=cv_stds, capsize=5, color='orange', alpha=0.7)
        axes[1,0].set_title('Cross-Validation Scores', fontweight='bold')
        axes[1,0].set_ylabel('CV Score')
        axes[1,0].set_ylim(0, 1)
        
        # 4. Results table
        axes[1,1].axis('tight')
        axes[1,1].axis('off')
        table_data = []
        for name in names:
            table_data.append([
                name,
                f"{self.results[name]['accuracy']:.3f}",
                f"{self.results[name]['f1']:.3f}",
                f"{self.results[name]['cv_mean']:.3f} (+/- {self.results[name]['cv_std']:.3f})"
            ])
        
        table = axes[1,1].table(cellText=table_data,
                                colLabels=['Model', 'Accuracy', 'F1-Score', 'CV Score'],
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.suptitle('Model Comparison Results', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n📸 Model comparison saved as 'model_comparison.png'")
        
        return best_model_name
    
    def interactive_cli(self):
        """Simple CLI without display issues"""
        print("\n" + "="*60)
        print("🚀 SENTIMENT ANALYSIS TOOL")
        print("="*60)
        print("\nCommands:")
        print("  'analyze: <text>' - Analyze sentiment")
        print("  'test' - Run test samples")
        print("  'stats' - Show model statistics")
        print("  'exit' - Quit")
        
        # Get best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        pipeline = self.results[best_model_name]['pipeline']
        print(f"\n✅ Using best model: {best_model_name}")
        
        while True:
            try:
                user_input = input("\n>> ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'stats':
                    print("\n📊 MODEL STATISTICS")
                    print("="*40)
                    for name, results in self.results.items():
                        print(f"\n{name}:")
                        print(f"  Accuracy: {results['accuracy']:.4f}")
                        print(f"  F1-Score: {results['f1']:.4f}")
                        print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
                
                elif user_input.lower() == 'test':
                    test_texts = [
                        "I absolutely love this product! It's amazing!",
                        "This is terrible, worst purchase ever",
                        "It's okay, nothing special but works",
                        "Fantastic quality and excellent service!",
                        "Very poor quality, completely disappointed"
                    ]
                    
                    print("\n📝 TEST RESULTS")
                    print("="*50)
                    for i, text in enumerate(test_texts, 1):
                        clean_text = self.advanced_preprocessing(text)
                        pred = pipeline.predict([clean_text])[0]
                        proba = pipeline.predict_proba([clean_text])[0]
                        
                        sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
                        confidence = max(proba)
                        
                        print(f"\n{i}. Text: {text}")
                        print(f"   Sentiment: {sentiment}")
                        print(f"   Confidence: {confidence:.2f}")
                
                elif user_input.startswith('analyze:'):
                    text = user_input[8:].strip()
                    if text:
                        clean_text = self.advanced_preprocessing(text)
                        pred = pipeline.predict([clean_text])[0]
                        proba = pipeline.predict_proba([clean_text])[0]
                        
                        sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
                        print(f"\n📊 Result:")
                        print(f"Text: {text}")
                        print(f"Sentiment: {sentiment}")
                        print(f"Confidence: {max(proba):.2f}")
                        print(f"Positive: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
                
                elif user_input:
                    # Direct analysis
                    clean_text = self.advanced_preprocessing(user_input)
                    pred = pipeline.predict([clean_text])[0]
                    proba = pipeline.predict_proba([clean_text])[0]
                    
                    sentiment = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
                    print(f"\n📊 Result: {sentiment} (confidence: {max(proba):.2f})")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def main():
    print("🔧 Initializing Sentiment Analyzer...")
    
    # Initialize
    analyzer = ProfessionalSentimentAnalyzer()
    
    # Load/download data
    print("\n📂 Loading dataset...")
    df = analyzer.download_dataset()
    
    # Explore data (saves plots instead of showing)
    df = analyzer.explore_data(df)
    
    # Preprocess
    print("\n🔄 Preprocessing text...")
    df['clean_text'] = df['text'].apply(analyzer.advanced_preprocessing)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['sentiment'], 
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    print(f"\n📊 Training samples: {len(X_train)}")
    print(f"📊 Testing samples: {len(X_test)}")
    
    # Train multiple models
    results = analyzer.train_multiple_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    best_model = analyzer.compare_models(y_test)
    print(f"\n🏆 Best Model: {best_model}")
    
    # Save best model
    joblib.dump(results[best_model]['pipeline'], 'sentiment_model.pkl')
    print("💾 Model saved as 'sentiment_model.pkl'")
    
    print("\n✅ Setup complete! Check the generated PNG files for visualizations.")
    
    # Start CLI
    analyzer.interactive_cli()

if __name__ == "__main__":
    main()