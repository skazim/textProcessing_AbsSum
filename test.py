import pandas as pd
import re
from contractions import contractions_dict


def data_preprocess(df1,df2):
    df1_columns = df1.columns.tolist()
    df1_columns.remove('headlines')
    df1_columns.remove('text')
    df1.drop(df1_columns, axis='columns', inplace=True)

    df = pd.concat([df1, df2], axis='rows')
    del df1, df2
    df = df.sample(frac=1).reset_index(drop=True)
    df.text = df.text.apply(str.lower)
    df.headlines = df.headlines.apply(str.lower)
    return df

def expand_contractions(text, contraction_map=contractions_dict):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            print(match)
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data['ctext'], data['headlines']


filename1 = 'news_summary.csv'
filename2 = 'news_summary_more.csv'

df1 = pd.read_csv(filename1, encoding='iso-8859-1').reset_index(drop=True)
df2 = pd.read_csv(filename2, encoding='iso-8859-1').reset_index(drop=True)

df = data_preprocess(df1,df2)

headlines = data_preprocess(df1,df2)['headlines']
text = data_preprocess(df1,df2)['text']
print(headlines.sample(5))
print(text.sample(5))

print(expand_contractions("y'all can't expand contractions i'd think"))
# file_path = 'example_dataset.csv'
# input_texts, target_texts = load_data(file_path)
# print(input_texts)