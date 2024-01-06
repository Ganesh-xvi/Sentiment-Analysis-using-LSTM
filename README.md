
# Sentiment Analysis using LSTM

This GitHub repository contains code for sentiment analysis using Long Short-Term Memory (LSTM) neural networks. The project focuses on analyzing McDonald's customer reviews to classify sentiments as positive, negative, or neutral. The sentiment analysis model is implemented in Python using TensorFlow and Keras.




## Project Overview

The project involves the following key components:

- Data Preprocessing: Cleaning and transforming raw text data, handling emojis, and converting ratings to sentiment labels.
- Text Processing: Tokenizing, removing stopwords using SpaCy, and creating sequences for model input.
- Embedding Layer: Utilizing pre-trained word embeddings (GloVe) for improved model performance.
- LSTM Model: Implementing a Bidirectional LSTM model with dropout layers for sentiment classification.
- Model Training: Training the LSTM model on the processed data with early stopping to prevent overfitting.
- Evaluation: Assessing the model's accuracy and performance metrics on a test dataset.
- Predictions: Making sentiment predictions on sample positive and negative reviews.

## Results

The LSTM model achieves promising accuracy in sentiment classification, demonstrating its effectiveness in understanding and categorizing McDonald's customer sentiments.
## Future Work

- Integration with real-world customer feedback systems.
- Fine-tuning the model for domain-specific sentiment analysis tasks.
Feel free to explore the Jupyter notebook (McDonalds_Sentiment_Analysis.ipynb) for a step-by-step walkthrough of the sentiment analysis process.
## Dataset Link

https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews