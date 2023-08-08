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


def bag_of_words(train_set, operation="train"):
    sms_list = [sms for sms in train_set['SMS']]
    vector = CountVectorizer()
    matrix = vector.fit_transform(sms_list)
    bag_of_words_df = pd.DataFrame(matrix.toarray(), columns=vector.get_feature_names_out())
    return bag_of_words_df


def main():
    transform_data()
    df_random = df.sample(frac=1, random_state=43)
    df_random.reset_index(drop=True, inplace=True)
    train_last_index = int(df_random.shape[0] * 0.8)
    train_set = df_random[0:train_last_index]

    train_bag_of_words = bag_of_words(train_set)
    train_bag_of_words.reset_index(drop=True, inplace=True)
    train_set = pd.concat([train_set, train_bag_of_words], axis=1)

    pd.options.display.max_columns = train_set.shape[1]
    pd.options.display.max_rows = train_set.shape[0]
    print(train_set.iloc[:200, :50])


if __name__ == '__main__':
    main()
