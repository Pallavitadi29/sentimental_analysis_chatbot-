# sentimental_analysis_chatbot-
1.Project Description
Sentiment analysis is the process of determining the sentiment (positive, negative, neutral) expressed in a piece of text. This project uses the Naïve Bayes algorithm to classify the sentiment of input text and displays the results using a user-friendly Streamlit app.

2.Project Motivation
Sentiment analysis is a fundamental task in natural language processing that has various real-world applications. Understanding the sentiment expressed in text data can provide valuable insights into user opinions, emotions, and trends. This project was motivated by the desire to explore sentiment analysis techniques and showcase their implementation through an interactive web application.

3.The goals of this project include:

Demonstrating how the Naïve Bayes algorithm can be used for sentiment classification.
Creating an intuitive Streamlit chatbot for users to easily interact with the sentiment analysis tool.
Showcasing text preprocessing techniques to enhance the accuracy of sentiment predictions.
Providing a practical example of using machine learning in real-world scenarios.
By sharing this project, we aim to contribute to the knowledge and understanding of sentiment analysis while providing a hands-on example for those interested in exploring natural language processing and interactive web application development.

4.The dataset used for this project is the "Tweet Sentiment Extraction" dataset from Kaggle. This dataset contains tweets along with their associated sentiment labels and selected text. The selected text provides a concise representation of the tweet's sentiment. The dataset is utilized to train sentiment analysis models for predicting the sentiment of tweets.

5.Columns
textID: A unique ID for each piece of text.
text: The text of the tweet.
sentiment: The general sentiment label of the tweet (positive, negative, or neutral).
selected_text (Training only): The text that supports the tweet's sentiment, serving as a sentiment indicator.
Dataset Problem statement
Given the text of a tweet, the task is to classify the sentiment as positive, negative, or neutral. This involves training a model to understand the emotional tone of the text.

6.Project Directory Structure
│                      
├── app.py                           # Streamlit application script
├── data                             # Directory for storing the dataset
│   └── train.csv                    # Sentiment dataset
├── images                           # Directory for sentiment image
│   ├── app_Sentiment_1.jpg          # web app screenshot 1
│   └── app_Sentiment_2.jpg          # web app screenshot 2
│   └── app_Sentiment_3.jpg          # web app screenshort 3
│   └── negative.jpg                 # Positive sentiment image
│   └── neutral.jpg                  # Positive sentiment image
│   └── positive.jpg                 # Positive sentiment image
│   └── sentimentanalysishotelgeneric-2048x803-1.jpg
├── docs                             # documentation for your project
├── .gitignore                       # ignore files that cannot commit to Git
├── notebooks                        # store notebooks
│   └── EDA_sentiment_analysis.ipynb # EDA Notebook
├── logs.txt                         # Streamlit log files 
├── requirements.txt                 # List of required packages
├── README.md                        # Project README file
Description
Sentiment analysis is the process of determining the sentiment (positive, negative, neutral) expressed in a piece of text. This project uses the Naïve Bayes algorithm to classify the sentiment of input text and displays the results using a user-friendly Streamlit app.

7.Features
Preprocesses text data (lowercase, punctuation removal, etc.).
Calculates word counts and likelihoods for sentiment classification.
Displays sentiment classification results with sentiment scores.
Displays resized sentiment-specific images based on the predicted sentiment.
Provides a visually appealing layout for user interaction.

8.Data Collection and Preprocessing:

Gather a dataset containing positive, negative, and neutral sentiment-labeled text.
Preprocess the text data by converting to lowercase, removing punctuation, and tokenizing sentences.
Calculating Word Counts and Likelihoods:

Create word count tables for each sentiment class.
Calculate the likelihood of each word appearing in a specific sentiment class using Laplace smoothing.
Train-Test Split and Model Training:

Shuffle the dataset and split it into training and testing subsets.
Train the Naïve Bayes model using the training data and calculated likelihoods.
Creating the Streamlit App:

Build a Streamlit web application for user interaction.
Incorporate text input, sentiment classification, and display of sentiment scores.
Display sentiment-specific images based on the predicted sentiment.
Running the App:

Install the required packages using pip install streamlit pandas nltk.
Run the Streamlit app using streamlit run app.py.
Interact with the App:

Enter text in the provided text area and click the "Classify Sentiment" button.
View the predicted sentiment label, sentiment scores, and corresponding image.
Requirements
Python 3.x
Streamlit
Pandas
NLTK (Natural Language Toolkit)















Notebook Structure
The Jupyter Notebook (EDA_sentiment_analysis.ipynb) is structured as follows:

Introduction and Setup: Importing libraries and loading the dataset.
Data Exploration: Displaying basic dataset information.
Sentiment Distribution Visualization: Visualizing the distribution of sentiment labels.
Text Preprocessing: Defining preprocessing functions for tokenization and stemming.
Word Count Analysis: Calculating word counts for different sentiment classes.
Top Words Visualization: Displaying top words for each sentiment class and creating treemap visualizations.
Running the Notebook
Follow these steps to run the EDA_sentiment_analysis.ipynb notebook:

Ensure you have Python and the required libraries installed.
Open the notebook using Jupyter Notebook or Jupyter Lab.
Execute each cell sequentially to see the analysis results.
Results and Visualizations
The notebook produces various insightful visualizations, including:

Sentiment distribution using a Funnel-Chart.
Top words and their counts for positive, negative, and neutral sentiments.
Treemap visualizations of top words for each sentiment class.
Sample images of these visualizations are provided in the repository's images folder.
