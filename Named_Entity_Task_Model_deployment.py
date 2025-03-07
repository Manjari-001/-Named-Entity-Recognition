#!/usr/bin/env python
# coding: utf-8

# # Data Preparation Module (data_preparation.py):

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(file_path):
    # Read the CSV file containing sentences and tags
    ner_df = pd.read_csv(file_path, encoding="latin1")
    return ner_df

def split_data(ner_df):
    # Split the dataset into train, validation, and test sets
    train_df, test_df = train_test_split(ner_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    return train_df, val_df, test_df

def preprocess_data(df):
    # Function to preprocess the data
    sentences = []
    ner_tags = []
    current_sentence = []
    current_tags = []

    for index, row in df.iterrows():
        # Check if the Sentence # column is not NaN
        if not pd.isnull(row["Sentence #"]):
            # If it's not a blank sentence, add the current sentence and its tags to the lists
            if current_sentence:
                sentences.append(current_sentence)
                ner_tags.append(current_tags)
            # Reset current sentence and tags for the new sentence
            current_sentence = []
            current_tags = []
        # Add word and tag to the current sentence and tags
        current_sentence.append(row["Word"])
        current_tags.append(row["Tag"])
    
    # Append the last sentence and tags
    if current_sentence:
        sentences.append(current_sentence)
        ner_tags.append(current_tags)
    
    return sentences, ner_tags


# # Feature Extraction Module (feature_extraction.py): This module will contain functions for extracting features from sentences.

# In[11]:


def word_features(sentence, index):
    word = sentence[index]

    # Define features for the word
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }

    if index > 0:
        prev_word = sentence[index - 1]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
            '-1:word.isupper()': prev_word.isupper(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    
    if index < len(sentence) - 1:
        next_word = sentence[index + 1]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
            '+1:word.isupper()': next_word.isupper(),
        })
    else:
        features['EOS'] = True  # End of sentence
    
    return features


# In[4]:


def sentence_features(sentence):
    return [word_features(sentence, index) for index in range(len(sentence))]

def sentence_labels(sentence):
    return [label for label in sentence]


# # Model Training Module (model_training.py): This module will contain code for training the model.

# In[5]:


import sklearn_crfsuite

def train_crf(X_train, y_train):
    # Initialize CRF model
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',  # Optimization algorithm
        c1=0.1,             # L1 regularization coefficient
        c2=0.1,             # L2 regularization coefficient
        max_iterations=100, # Maximum number of iterations
        all_possible_transitions=True  # Allow transitions between all labels
    )

    # Train the CRF model
    crf.fit(X_train, y_train)
    return crf


# In[24]:


#pip install --upgrade tensorflow keras


# In[42]:


import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Bidirectional, TimeDistributed, Dense, Dropout, Masking, Lambda
from tensorflow.keras import backend as K

def train_model(train_sentences, val_sentences, train_ner_tags, val_ner_tags):
    # Tokenization
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(train_sentences)

    # Convert words to sequences of integers
    X_train_seq = word_tokenizer.texts_to_sequences(train_sentences)
    X_val_seq = word_tokenizer.texts_to_sequences(val_sentences)

    # Padding
    max_seq_length = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length, padding='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=max_seq_length, padding='post')

    # Label Encoding
    label_encoder = LabelEncoder()
    label_encoder.fit([tag for tag_seq in train_ner_tags for tag in tag_seq])

    # Encode NER tags
    y_train_encoded = [label_encoder.transform(tag_seq) for tag_seq in train_ner_tags]
    y_val_encoded = [label_encoder.transform(tag_seq) for tag_seq in val_ner_tags]

    # Define the input shape
    input_layer = Input(shape=(max_seq_length,))

    # Add an embedding layer to convert input words into dense vectors
    embedding_layer = Embedding(input_dim=len(word_tokenizer.word_index) + 1, output_dim=100, input_length=max_seq_length)(input_layer)

    # Add a Bidirectional LSTM layer to capture bidirectional context
    lstm_layer = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.5))(embedding_layer)

    # Add a TimeDistributed layer to apply a dense layer to each time step
    dense_layer = TimeDistributed(Dense(units=len(label_encoder.classes_), activation="softmax"))(lstm_layer)

    # Define the model
    model = Model(inputs=input_layer, outputs=dense_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(X_train_pad, np.array(y_train_encoded), validation_data=(X_val_pad, np.array(y_val_encoded)), epochs=5, batch_size=32, verbose=1)

    return model, label_encoder, word_tokenizer


# # Model Evaluation Module (model_evaluation.py): This module will contain code for evaluating the model.

# In[15]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

def evaluate_model(crf, X_val, y_val, y_val_pred):
    # Fit MultiLabelBinarizer on the combined set of labels from both true and predicted values
    mlb = MultiLabelBinarizer()
    mlb.fit(y_val + y_val_pred)

    # Transform true and predicted values into binary arrays
    y_val_bin = mlb.transform(y_val)
    y_val_pred_bin = mlb.transform(y_val_pred)

    # Evaluate the model's performance on the validation set
    report = classification_report(y_val_bin, y_val_pred_bin, target_names=mlb.classes_)
    return report


# In[33]:


import numpy as np

def evaluate(model, X_test_pad, y_test_encoded):
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test_pad, y_test_encoded, verbose=1)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Predict labels for the test data
    predictions = model.predict(X_test_pad)

    # Decode the predicted labels
    decoded_predictions = []

    for i in range(len(predictions)):
        # Get the predicted label indices for the current sequence
        predicted_indices = np.argmax(predictions[i], axis=1)

        # Decode the predicted label indices using the label encoder
        predicted_labels = label_encoder.inverse_transform(predicted_indices)

        # Add the decoded predictions to the list
        decoded_predictions.append(predicted_labels)

    # Print the first few decoded predictions
    for i in range(5):
        print("Sentence:", test_sentences[i])
        print("True Labels:", y_test[i])
        print("Predicted Labels:", decoded_predictions[i])
        print()



# # Prediction Module (prediction.py): This module will contain code for making predictions with the trained model.

# In[7]:


def predict_sentence(crf, sentence):
    # Preprocess the example sentence
    example_features = [sentence_features(sentence)]

    # Make predictions on the example sentence
    predicted_labels = crf.predict(example_features)[0]
    return predicted_labels


# In[32]:


import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict(model, tokenizer, label_encoder, test_sentences):
    # Convert words to sequences of integers
    X_test_seq = tokenizer.texts_to_sequences(test_sentences)

    # Padding
    max_seq_length = 100
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length, padding='post')

    # Predict labels for the test data
    predictions = model.predict(X_test_pad)

    # Decode the predicted labels
    decoded_predictions = []

    for i in range(len(predictions)):
        # Get the predicted label indices for the current sequence
        predicted_indices = np.argmax(predictions[i], axis=1)

        # Decode the predicted label indices using the label encoder
        predicted_labels = label_encoder.inverse_transform(predicted_indices)

        # Add the decoded predictions to the list
        decoded_predictions.append(predicted_labels)

    return decoded_predictions



# # Main file : Call all functions 

# In[1]:


#from data_preparation import read_data, split_data, preprocess_data

# Read the dataset
dataset_path = "ner_dataset.csv"
ner_df = read_data(dataset_path)

# Split the dataset
train_df, val_df, test_df = split_data(ner_df)

# Preprocess training, validation, and test data
train_sentences, train_ner_tags = preprocess_data(train_df)
val_sentences, val_ner_tags = preprocess_data(val_df)
test_sentences, test_ner_tags = preprocess_data(test_df)


# In[12]:


#Fetaure extraction 

#from feature_extraction import sentence_features, sentence_labels

# Extract features and labels for the training data
X_train = [sentence_features(sentence) for sentence in train_sentences]
y_train = [sentence_labels(sentence) for sentence in train_ner_tags]

# Extract features and labels for the validation data
X_val = [sentence_features(sentence) for sentence in val_sentences]
y_val = [sentence_labels(sentence) for sentence in val_ner_tags]

# Extract features and labels for the test data
X_test = [sentence_features(sentence) for sentence in test_sentences]
y_test = [sentence_labels(sentence) for sentence in test_ner_tags]


# In[14]:


#Model training 

#from model_training import train_crf

# Train the CRF model
crf = train_crf(X_train, y_train)


# In[17]:


#Model evaluation 

#from model_evaluation import evaluate_model

# Make predictions on the validation set
y_val_pred = crf.predict(X_val)

# Evaluate the model's performance on the validation set
report = evaluate_model(crf, X_val, y_val, y_val_pred)
print("CRF Model Evaluation on Validation Set:")
print(report)


# In[58]:


# The CRF model demonstrates strong precision and recall for most named entity categories, particularly achieving 
# high scores for B-geo, B-gpe, B-org, B-per, and O labels. However, it struggles with less frequent entities such 
# as B-art, B-eve, B-nat, I-art, I-eve, and I-nat, where precision and recall are relatively low. Overall, the model
# performs well on the validation set, with an F1-score of 0.86, indicating its effectiveness in capturing named 
# entities across various categories.


# In[18]:


#Prediction model 

#from prediction import predict_sentence

# Select an example sentence from the validation set
example_sentence = val_sentences[0]

# Make predictions on the example sentence
predicted_labels = predict_sentence(crf, example_sentence)
print("Predicted Labels:", predicted_labels)


# In[56]:


from sklearn.preprocessing import LabelEncoder

# Fit label encoder on the combined set of true and predicted values
label_encoder = LabelEncoder()
all_labels = y_val + y_val_pred
label_encoder.fit([label for sequence in all_labels for label in sequence])

decoded_predictions = []

# Decode the predictions
for sequence in y_val_pred:
    # Get the predicted label indices for the current sequence
    predicted_indices = label_encoder.transform(sequence)
    # Decode the predicted label indices using the label encoder
    predicted_labels = label_encoder.inverse_transform(predicted_indices)
    # Add the decoded predictions to the list
    decoded_predictions.append(predicted_labels)

# Print the first few decoded predictions
for i in range(min(5, len(val_sentences))):
    print("Sentence:", val_sentences[i])
    print("True Labels:", y_val[i])
    print("Predicted Labels:", decoded_predictions[i])
    print()



# In[57]:


# The model shows consistent accuracy in recognizing entities like geopolitical entities (B-gpe), organizations 
# (B-org), and persons (B-per). However, it occasionally misclassifies entities, such as identifying 'International' 
# as an organization instead of a geopolitical entity and 'Luzhkov' as a person instead of an organization. 
# Additionally, it struggles with lower-frequency entities. Despite these minor discrepancies, the model performs reasonably 
# well overall in identifying named entities across various sentences.

