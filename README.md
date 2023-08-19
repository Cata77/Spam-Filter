# Spam Filter

The **Spam Filter Project with Naive Bayes Algorithm** is an intelligent solution designed to automatically categorize 
incoming text messages as either spam or legitimate (ham) messages. By employing the Naive Bayes algorithm, a probabilistic 
approach that leverages the occurrence of words within messages, the system efficiently discerns between undesirable spam 
content and genuine communication.

## Importing Data

The project starts by loading [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) 
from UCI Machine Learning repository containing labeled text messages into a Pandas DataFrame. The dataset's columns are 
renamed to 'Target' (for spam or ham label) and 'SMS' (for the text content of the messages). 

## Data Preprocessing

The text preprocessing phase begins by converting all text to lowercase, removing punctuation, and replacing numerical digits with a placeholder ('aanumbers') to 
focus solely on the semantic content of words. Furthermore, the text is lemmatized using the spaCy library to reduce words to 
their base forms, and stopwords are eliminated to eliminate common, non-discriminatory terms.

The feature extraction process involves transforming the preprocessed text data into a bag-of-words representation. 
The CountVectorizer class from Scikit-learn is employed to convert text into numerical feature vectors. 
This matrix represents the frequency of each word in the messages across the entire dataset.

## Data Processing

The project calculates probabilities for each word in the vocabulary with respect to spam and ham categories. 
Laplace smoothing is applied to avoid zero probabilities, ensuring that even previously unseen words contribute to the 
classification process. By summing the word counts for both spam and ham messages and utilizing a set of formulae, 
the likelihood of a word appearing in spam or ham messages is determined.

Classification is performed on new incoming messages. Each message is tokenized and processed word by word. 
The probabilities calculated earlier are utilized to estimate the probability of the message being spam or ham, based on 
the words present. The Naive Bayes assumption that word occurrences are independent is applied here. 
Finally, the message is assigned a classification label of 'spam,' 'ham,' or 'unknown,' depending on whether the calculated 
spam probability surpasses the ham probability.

## Naive Bays classifier

The Naive Bayes classifier is a probabilistic model based on the Bayes theorem. It's one of the easiest classification models.
A Naive Bayes classifier considers that each of the features independently contributes to the probability whether the given 
text is a spam or not. So, we call this classifier naive because it processes features independently, and this is what makes
it somewhat similar to the bag-of-words model. It also assumes the independence of words in the model.

The conditional probabilities can be calculated with the equation below:
$$P(word_i|spam) = \frac{N_{word_i|spam}}{N_{spam} + N_{vocab}}$$

$N_{word_i|spam}$ is the number of **word_i** in the spam; $N_{spam}$ is the total number of words in spam; $N_{vocab}$ is the total
number of words in the training set vocabulary

$$P(word_i|ham) = \frac{N_{word_i|ham}}{N_{ham} + N_{vocab}}$$

$N_{word_i|ham}$ is the number of $**word_i**$ in the ham; $N_{ham}$ is the total number of words in ham.

You can take the list of words from the spam and ham subsets of the training set. Some words from ham do not occur in spam 
and vice versa. These words are considered <b>zero-probability words</b>. If a zero-probability word from the spam subset 
occurs in a sentence of the test set, the spam conditional probability will be zero. To solve this problem, we will 
introduce the <b>Laplace smoothing </b> ($\alpha$). In this project, we will assume that the Laplace smoothing is 1.

The conditional probability of words in the spam subset can be adjusted as follows:
$$P(word_i|spam) = \frac{N_{word_i|spam} + \alpha}{N_{spam} + \alpha N_{vocab}}$$

while the conditional probability of words in the ham subset becomes:
$$P(word_i|ham) = \frac{N_{word_i|ham} + \alpha}{N_{ham} + \alpha N_{vocab}}$$

To get the probability of whether a given list of words is spam, we need to multiply the conditional probabilities of each 
word in the sentence by the probability of spam occurring in the dataset. 

This conditional probability of spam in a list of words is shown below:
$$P(spam|word_1, word_2, ..., word_n) = P(word_1|spam) \times P(word_2|spam) \times...\times P(word_n|spam)$$

Similarly, the conditional probability of ham in a given list of words can be calculated as follows:
$$P(ham|word_1, word_2, ..., word_n) = P(word_1|ham) \times P(word_2|ham) \times...\times P(word_n|ham)$$

Now, we would like to discuss **P(spam)** and **P(ham)** probabilities. 
**P(spam)** is the probability of spam occurrence in the training dataset. It is the number of rows in the target column 
labeled as spam divided by the total number of rows. In terms of mathematics, it can be represented as follows:
$$P(spam) = \frac{N_{spam}}{N_{spam} + N_{ham}}$$

Similarly, **P(ham)** can be represented as:
$$P(ham) = \frac{N_{ham}}{N_{spam} + N_{ham}}$$

To classify a message as spam or ham in sentence, calculate $P(spam|word_1, word_2, ..., word_n)$ and 
P(ham|word_1, word_2, ..., word_n), and then compare their values. If:

$P(spam|word_1, word_2, ..., word_n) > P(ham|word_1, word_2, ..., word_n)$ the sentence is classified as **spam**, but if:
$P(spam|word_1, word_2, ..., word_n) < P(ham|word_1, word_2, ..., word_n)$ the sentence is classified as **ham**. If the 
values are equal, the sentence is classsified as **unknown**, for manual reviewing.

## Performance results

The project evaluates its performance using various metrics, including accuracy, recall, precision, and the F1 score. 
Confusion matrix analysis is conducted to assess how well the algorithm is identifying true positives, true negatives, 
false positives, and false negatives.

A confusion matrix is a table that describes the performance of a classification model. 
This is an example of a confusion matrix:
| True Positive | False Positive |
|---------------|----------------|
| False Negative| True Negative  |

In this classification, the spam predicted as spam is **TP**, and the ham predicted as ham is **TN**. 
Classification errors are presented in **FP** and **FN**.

$$Accuracy = \frac{TP\ +\ TN}{TP\ +\ TN\ +\ FP\ +\ FN}$$

$$Precision = \frac{TP}{TP\ +\ FP}$$

$$Recall = \frac{TP}{TP\ +\ FN}$$

$$F-measure = \frac{2 \times Precision \times Recall}{Precision\ +\ Recall}$$

In the previous stage, I have successfully built a working Multinomial Naive Bayes classification model from scratch. 
It would be a good idea to compare the result with another model. The popular **Scikit-Learn library** also contains an 
implementation of the Multinomial Naive Bayes classifier, so we will compare how our model performs against it.

## Conclusions

The project exemplifies a practical application of natural language processing and machine learning techniques to address 
a common problem of distinguishing spam from legitimate messages. It showcases the Naive Bayes algorithm's effectiveness 
in probabilistic classification, the importance of preprocessing in text analysis, and the significance of thorough 
evaluation to measure model performance accurately. 

Through its well-structured steps and thoughtful implementation, the project serves as a valuable resource for those 
seeking to develop similar spam filtering systems using the Naive Bayes algorithm.

