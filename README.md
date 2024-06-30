# SMS Classifier
## Overview
This project involves developing a text classification model to classify SMS messages as either spam or non-spam using data science techniques in Python. The model leverages Natural Language Processing (NLP) techniques and the Scikit-learn library to preprocess text data, transform it into numerical representations, and train a machine learning model for classification.

## Dataset
The dataset used in this project contains SMS messages labeled as either 'ham' (non-spam) or 'spam'. It is loaded from a publicly available source and consists of two columns:
the label indicates whether the SMS is 'ham' or 'spam'.

message: The text content of the SMS.
## Dependencies
The project requires the following Python libraries:

pandas

numpy

scikit-learn

## Implementation
The implementation involves the following steps:

Loading the Dataset:

The dataset is loaded from a URL and read into a pandas DataFrame.
## Preprocessing:

The labels are mapped to binary values (0 for ham and 1 for spam).
## Splitting the Data:

The data is split into training and test sets using train_test_split from Scikit-learn.
## Pipeline Creation:

A pipeline is created to streamline the process of transforming the text data and training the model. 

This pipeline includes:
CountVectorizer: Converts the text data into a matrix of token counts.

TfidfTransformer: Transforms the count matrix into a normalized tf-idf representation.

MultinomialNB: A Naive Bayes classifier for multinomially distributed data.
## Model Training and Prediction:

The pipeline is fitted to the training data and then used to make predictions on the test data.
## Model Evaluation:

The model's performance is evaluated using accuracy, a confusion matrix, and a classification report.
## Results
After running the implementation, you will obtain the following outputs:

Accuracy: The accuracy of the model on the test data.

Confusion Matrix: A matrix showing the number of correct and incorrect predictions.

Classification Report: A detailed report showing each class's precision, recall, and F1-score.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

Alolika Bhowmik

alolikabhowmik72@gmail.com
