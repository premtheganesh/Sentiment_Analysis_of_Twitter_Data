## Sentiment Analysis Using Twitter Data
<br>
## Introduction
Social media, particularly Twitter, has become a primary medium for expressing opinions and reactions to real-world events. This project demonstrates how Natural Language Processing (NLP) techniques can be applied to analyze such data and classify tweets as positive or negative.
The dataset consists of 1.6 million pre-labeled tweets from Kaggle, containing both positive and negative sentiments. The project walks through the end-to-end pipeline of data collection, cleaning, feature extraction, model training, and evaluation.
## Problem Statement
Twitter data is unstructured, noisy, and high-volume. Tweets often contain slang, hashtags, emojis, and abbreviations that make them challenging for traditional analysis.
The key challenges addressed in this project include:
* Cleaning and preprocessing raw tweets.
* Handling contractions, misspellings, and unwanted characters.
* Converting text into numerical vectors for machine learning.
* Building and comparing classification models.
* Evaluating performance using standard metrics to identify the best approach.
## Dataset
Source: Kaggle Sentiment140 dataset (1.6M tweets).
* Columns used:
o Sentiment (0 = Negative, 4 = Positive)
o Tweet (raw tweet text)
The dataset was cleaned and transformed to contain only the sentiment labels and tweet text.
## Methodology
1. Data Preprocessing
* Text normalization: lowercasing, removing URLs, mentions, hashtags, numbers, and extra spaces.
* Handling contractions (e.g., "can't" ? "can not")
* Removing stopwords while preserving meaningful words.
* Lemmatization to reduce words to their root forms.
2. Exploratory Analysis
* Distribution of positive vs negative tweets.
* Word frequency plots to identify common terms.
* Word clouds for visual exploration of positive and negative sentiments.
3. Feature Engineering
Several representation methods were compared:
* Bag of Words (BoW): Basic frequency-based representation.
* TF-IDF: Weighted representation that emphasizes important but less frequent words.
* Word2Vec: Custom embeddings trained on the dataset, producing semantic vectors.
* FastText: Subword-based embeddings to capture misspellings and rare words.
* GloVe: Pretrained embeddings from large corpora, applied with weighted averaging.
* BERT: Contextual transformer-based embeddings, fine-tuned for this dataset.
4. Model Building
For each feature set, multiple models were trained and compared:
* Traditional ML models: Logistic Regression, Naive Bayes, Support Vector Machines (SVM), and Random Forest.
* Embedding-based models: Word2Vec, FastText, and GloVe combined with Logistic Regression, SVM, and Naive Bayes.
* Transformer-based model: Fine-tuned BERT using Hugging Face Transformers.
5. Evaluation
Models were assessed using:
* Accuracy
* Precision, Recall, and F1-score
* Confusion matrices for error analysis
## Results
* Traditional ML (BoW & TF-IDF)
o TF-IDF outperformed BoW, achieving ~79.7% accuracy.
o Logistic Regression and SVM delivered the strongest results among traditional classifiers.
* Word Embeddings (Word2Vec, FastText, GloVe)
o Word2Vec achieved ~77.8% accuracy, the best among embeddings.
o GloVe reached ~73% accuracy, while FastText struggled (~56?57%).
o Embeddings were competitive but did not surpass TF-IDF.
* BERT (Transformer Model)
o Fine-tuned BERT achieved ~92.7% accuracy, with precision (93.1%) and F1-score (92.6%).
o BERT significantly outperformed all other approaches, confirming the strength of transformer-based contextual models.
Final Conclusion: While TF-IDF with Logistic Regression provided a strong baseline, BERT clearly dominated with a 13 - 15% improvement, making it the best-performing model for this task.
## Gradio Application
As a final step, a?basic Gradio interface?was built to demonstrate the model in action. This allows users to enter custom text and instantly receive a sentiment prediction 
The Gradio app showcases how the trained model can be integrated into an interactive system, moving beyond experimentation to practical application.
## Tools and Libraries
The project uses the following technologies:
* Data Handling: pandas, numpy
* NLP: nltk, gensim, contractions
* Modeling: scikit-learn, Hugging Face Transformers
* Visualization: matplotlib, seaborn, wordcloud
* Interface: Gradio
## Project Structure
*  Final-Sentiment_Analysis_for_Twitter_Data.ipynb # Main notebook: data loading, preprocessing, modeling, and evaluation
* app.py #Gradio app for deploying the model 
* requirements.txt # List of dependencies to reproduce the project environment
* README.md # Documentation file
