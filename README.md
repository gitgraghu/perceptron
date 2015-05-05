## **Part I: Perceptron**

[1.] Training Data model:

    - python3 perceplearn.py TRAINING_FILE MODEL_FILE
    
 > eg: python3 perceplearn.py spam_training.txt spam.nb

[2.] Classify Input Data:

    - python3 percepclassify.py MODEL_FILE < TEST_FILE > OUTPUT_FILE
    
 > eg: python3 percepclassify.py spam.nb < spam_test.txt > spam.out

## **Part II: Part of Speech Tagger**

Features Used: Word, Surrounding Words (left and right), WordShape, Suffixes

[1.] Training Data Model

    - python3 postrain.py TRAINING_FILE MODEL -h DEV_FILE

> eg: python3 postrain.py train.pos pos.model -h dev.pos

[2.] Classifying Input Data:

    - python3 postag.py MODEL < TEST_FILE > OUTPUT_FILE

> eg: python3 postag.py pos.model < pos.blind.test > pos.test.out

## **Part III: Named Entity Recognition**

Features Used: Word, Surrounding Words (left and right), Wordshape, Pos tag, Surrounding POS tags (left and right)

[1.] Training Data Model

    - python3 nelearn.py TRAINING_FILE MODEL -h DEV_FILE

> eg: python3 nelearn.py ner.esp.train ner.model -h ner.esp.dev

[2.] Classifying Input Data:

    - python3 netag.py ner.model < TEST_FILE > OUTPUT_FILE

> eg: python3 netag.py ner.model < ner.esp.blind.test > ner.esp.test.out

## **Part IV:**

[1. ] What is the accuracy of your part-of-speech tagger?

- **Training Data Accuracy:**	98.1406%
- **Development Data Accuracy:**96.3955%

[2. ] What are the precision, recall and F-score for each of the named entity types for your named entity recognizer, and what is the overall F-score?

- Overall Accuracy: 94.7754%

| Class | Precision | Recall   | F-Score  |
|-------|-----------|----------|----------|
| B-PER | 0.812300  | 0.832242 | 0.822150 |
| I-PER | 0.828018  | 0.846332 | 0.837075 |
|       |           |          |          |
| B-MISC| 0.606770  | 0.523595 | 0.562123 |
| I-MISC| 0.501098  | 0.348623 | 0.411181 |
|       |           |          |          |
| B-LOC | 0.648983  | 0.745934 | 0.694089 |
| I-LOC | 0.530516  | 0.670623 | 0.592398 |
|       |           |          |          |
| B-ORG | 0.806594  | 0.748235 | 0.776319 |
| I-ORG | 0.741610  | 0.485358 | 0.586725 |
|       |           |          |          |
|   O   | 0.982166  | 0.994510 | 0.988299 |


[3. ] What happens if you use your Naive Bayes classifier instead of your perceptron classifier (report performance metrics)? Why do you think that is?

- Naive Bayes Classifier for POS Tagging: The Accuracy is **92.8608%** which is about 4% lesser than perceptron classification accuracy.
- Naive Bayes Classifier takes about 8-9 seconds to classify the data, while perceptron takes about 1-2 seconds.
- On the other hand, the Naive Bayes Classifier is much faster to train. Perceptron on the other hand is computationally expensive to train as it goes through a number of iterations on the training data.

