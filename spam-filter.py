import re
from string import punctuation

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('spam.csv', encoding='iso-8859-1', usecols=['v1', 'v2'])
df.rename(columns={'v1': 'Target', 'v2': 'SMS'}, inplace=True)
nlp = spacy.load("en_core_web_sm")


def lemmatize_text(text):
    doc = nlp(text)
    new_text = [word.lemma_ for word in doc]
    return ' '.join(new_text)


def replace_digits_with_aanumbers(text):
    pattern = r'\b(?:\d+\w*|\w*\d+\w*)\b'
    result = re.sub(pattern, 'aanumbers', text)
    return result


def remove_spacy_stopwords(text):
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop and len(token) > 1]
    return ' '.join(filtered_words)


def transform_data():
    for i in df.index:
        df.at[i, 'Target'] = df.at[i, 'Target'].lower()
        df.at[i, 'SMS'] = df.at[i, 'SMS'].lower()
        df.at[i, 'SMS'] = lemmatize_text(df.at[i, 'SMS'])
        df.at[i, 'SMS'] = df.at[i, 'SMS'].translate(str.maketrans('', '', punctuation))
        df.at[i, 'SMS'] = replace_digits_with_aanumbers(df.at[i, 'SMS'])
        df.at[i, 'SMS'] = remove_spacy_stopwords(df.at[i, 'SMS'])


def bag_of_words(train_set, category, operation="train"):
    spam_sms_texts = train_set[train_set['Target'] == category]['SMS']
    vector = CountVectorizer()
    spam_word_counts = vector.fit_transform(spam_sms_texts)
    sum_word_counts_spam = spam_word_counts.sum(axis=0)
    vocabulary = vector.get_feature_names_out()
    word_count_spam = dict(zip(vocabulary, sum_word_counts_spam.tolist()[0]))
    return word_count_spam


def find_spam_and_ham_probability(train_set):

    alpha = 1  # Laplace smoothing parameter

    vector = CountVectorizer()
    all_sms_texts = train_set['SMS']
    vector.fit(all_sms_texts)

    # Transform both spam and ham SMS texts into matrices of word counts
    spam_word_counts = vector.transform(train_set[train_set['Target'] == 'spam']['SMS'])
    ham_word_counts = vector.transform(train_set[train_set['Target'] == 'ham']['SMS'])

    # Sum the word counts across all documents to get the count of each word
    sum_word_counts_spam = spam_word_counts.sum(axis=0)
    sum_word_counts_ham = ham_word_counts.sum(axis=0)
    total_spam_words = spam_word_counts.sum()
    total_ham_words = ham_word_counts.sum()

    # Get the vocabulary from the vector
    vocabulary = vector.get_feature_names_out()

    # Create a dictionary to store word counts for spam and ham
    word_count_spam = dict(zip(vocabulary, sum_word_counts_spam.tolist()[0]))
    word_count_ham = dict(zip(vocabulary, sum_word_counts_ham.tolist()[0]))

    spam_probability_list = []
    ham_probability_list = []

    for word in vocabulary:  # Iterate over the vocabulary
        spam_count = word_count_spam.get(word, 0)  # Get the count of the word in spam (default to 0 if not found)
        ham_count = word_count_ham.get(word, 0)  # Get the count of the word in ham (default to 0 if not found)

        spam_probability = (spam_count + alpha) / (total_spam_words + alpha * len(vocabulary))
        ham_probability = (ham_count + alpha) / (total_ham_words + alpha * len(vocabulary))

        spam_probability_list.append(spam_probability)
        ham_probability_list.append(ham_probability)

    probability_df = pd.DataFrame({'Spam Probability': spam_probability_list, 'Ham Probability': ham_probability_list},
                                  index=vocabulary)
    return probability_df


def main():
    transform_data()
    df_random = df.sample(frac=1, random_state=43)
    df_random.reset_index(drop=True, inplace=True)
    train_last_index = int(df_random.shape[0] * 0.8)
    train_set = df_random[0:train_last_index]

    df_random = find_spam_and_ham_probability(train_set)
    pd.options.display.max_columns = df_random.shape[1]
    pd.options.display.max_rows = df_random.shape[0]
    print(df_random.head(200))


if __name__ == '__main__':
    main()
