**About the Dataset**
The dataset consists of a file ner_dataset.csv that maps:

Sentence #: The sentence number.
Word: The first word of the respective sentence.
POS: The Part-of-Speech (POS) tag for each word (ignored for this case study).
Tag: The Named Entity Recognition (NER) tag for each word.

Each sentence is structured as a sequence of words with their corresponding NER tags.
Sample Sentence Mapping:
Today O  
Micheal B-PER  
Jackson I-PER  
and O  
Mark B-PER  
ate O  
lasagna O  
at O  
New B-geo  
Delhi I-geo  
. O  

Sequence Tagging Scheme: IOB2
I (Inside): The word is inside a named entity chunk.
O (Outside): The word does not belong to any named entity chunk.
B (Beginning): The word is the beginning of a named entity chunk.

**Dataset Columns:**
Sentence #: The sentence number.
Word: The word to be classified.
POS: The POS tags for respective words (ignored for this study).
Tag: The NER tags for respective words.


**Probable Tasks**
(Below pointers provide a direction for the case study.)
Data Splitting:
Divide the dataset into three parts:
Train
Validation
Test (at least 20%)
Evaluation Metrics:
Identify appropriate metrics for evaluating the model's performance.
Data Preprocessing:
Map words in each sentence to their respective NER tags.
Baseline Model Development:
Create a simple model that takes a sentence (list of words) as input and predicts NER tags for each word.
Model Improvement:
Identify shortcomings of the baseline model.
Develop a new model that overcomes these shortcomings.

**Future Scope:**
Identify ways to further optimize the model for better performance.
