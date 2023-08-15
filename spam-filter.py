import re
from string import punctuation

import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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


def bag_of_words(data_set, vector):
    train_bow = vector.transform(data_set['SMS'])
    data_df = pd.DataFrame(train_bow.toarray(), columns=vector.get_feature_names_out())
    return data_df


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


def classify_sentence(row, probability_set, spam_count, ham_count):
    sms_list = row['SMS'].split()
    spam_probability = spam_count / (spam_count + ham_count)
    ham_probability = ham_count / (spam_count + ham_count)

    for word in sms_list:
        if word in probability_set.index:
            spam_probability *= probability_set.loc[word]['Spam Probability']
            ham_probability *= probability_set.loc[word]['Ham Probability']

    predicted = 'unknown'
    if spam_probability > ham_probability:
        predicted = 'spam'
    elif spam_probability < ham_probability:
        predicted = 'ham'

    return pd.Series({'Predicted': predicted, 'Actual': row['Target']})


def calculate_confusion_matrix(dataframe):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in dataframe.index:
        if dataframe.at[i, 'Predicted'] == 'spam' and dataframe.at[i, 'Actual'] == 'spam':
            TP += 1
        elif dataframe.at[i, 'Predicted'] == 'ham' and dataframe.at[i, 'Actual'] == 'ham':
            TN += 1
        elif dataframe.at[i, 'Predicted'] == 'spam' and dataframe.at[i, 'Actual'] == 'ham':
            FP += 1
        elif dataframe.at[i, 'Predicted'] == 'ham' and dataframe.at[i, 'Actual'] == 'spam':
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = 2 * precision * recall / (precision + recall)
    performance_results = {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': F1}

    return performance_results


def convert_column_to_binary(dataframe):
    dataframe['Target'] = (dataframe['Target'] == 'spam').astype('int')
    return dataframe


def model_performance(train_set, test_set):
    vector = CountVectorizer()
    vector.fit(train_set['SMS'])
    train_bow = bag_of_words(train_set, vector)
    test_bow = bag_of_words(test_set, vector)

    train_set = convert_column_to_binary(train_set)
    test_set = convert_column_to_binary(test_set)

    model = MultinomialNB()
    X_train = train_bow.values
    y_train = train_set['Target'].values
    model.fit(X_train, y_train)

    X_test = test_bow.values
    y_test = test_set['Target'].values
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    performance_results = {'Accuracy': accuracy, 'Recall': recall, 'Precision': precision, 'F1': f1}

    return performance_results


def main():
    transform_data()
    df_random = df.sample(frac=1, random_state=43)
    df_random.reset_index(drop=True, inplace=True)
    train_last_index = int(df_random.shape[0] * 0.8)
    train_set = df_random[0:train_last_index]
    test_set = df_random[train_last_index:]
    test_set_model = pd.DataFrame(test_set)

    probability_df = find_spam_and_ham_probability(train_set)
    spam_count = test_set[test_set['Target'] == 'spam'].shape[0]
    ham_count = test_set[test_set['Target'] == 'ham'].shape[0]
    classify_df = test_set.apply(classify_sentence, axis=1, args=(probability_df, spam_count, ham_count))

    test_set = test_set.drop(columns=['Target'])
    result_df = pd.concat([test_set, classify_df], axis=1)
    result_df.columns = ['Target', 'Predicted', 'Actual']
    result_df = result_df.reset_index(drop=True)

    pd.options.display.max_columns = probability_df.shape[1]
    pd.options.display.max_rows = probability_df.shape[0]
    print(probability_df.head(200))

    print('\n------------------------------------------------------------------------\n')
    pd.options.display.max_columns = result_df.shape[1]
    pd.options.display.max_rows = result_df.shape[0]
    print(result_df.head(200))

    print('\n------------------------------------------------------------------------\n')
    classify_df = classify_df.reset_index()
    print(calculate_confusion_matrix(classify_df))

    print('\n------------------------------------------------------------------------\n')
    print('Model performance:\n', model_performance(train_set, test_set_model))


if __name__ == '__main__':
    main()
