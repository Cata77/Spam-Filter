import re
from string import punctuation

import pandas as pd
import spacy

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


for i in df.index:
    df.at[i, 'Target'] = df.at[i, 'Target'].lower()
    df.at[i, 'SMS'] = df.at[i, 'SMS'].lower()
    df.at[i, 'SMS'] = lemmatize_text(df.at[i, 'SMS'])
    df.at[i, 'SMS'] = df.at[i, 'SMS'].translate(str.maketrans('', '', punctuation))
    df.at[i, 'SMS'] = replace_digits_with_aanumbers(df.at[i, 'SMS'])
    df.at[i, 'SMS'] = remove_spacy_stopwords(df.at[i, 'SMS'])

pd.options.display.max_columns = df.shape[1]
pd.options.display.max_rows = df.shape[0]

print(df.head(200))
