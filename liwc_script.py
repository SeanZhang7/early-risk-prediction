import liwc
from tqdm import tqdm
from collections import defaultdict
import click
import pandas as pd
import os

@click.command()
@click.option('--data', help='path to csv or pickle file', required=True)
@click.option('--savedir', 'savedir', help='directory to where to save data', default='liwc_script_output')
@click.option('--save', 'savename', help='directory to where to save data', default='data')
@click.option('-c','--column', 'column', help='name of column of data to process liwc', multiple=False)
def main(data: str, column, savedir: str, savename: str):
    df = pd.read_csv(data) 

    """
    get LIWC 
    """
    df[column] = df[column].fillna("")
    liwc_tokens, word_tokens = process_sentences(df[column].values)
    df[f"{column}_tokens"] = word_tokens
    df[f"{column}_liwc_tokens"] = liwc_tokens

    new_col_keys = liwc.liwc_keys + ['POLITICAL', 'POSEMO', 'NEGEMO', 'FEAR', 'JOY', 'ANGER', 'SAD', 'PURPOSE', 'ORDER', 'JUSTICE']

    new_columns = {k:[] for k in new_col_keys}

    for v in df[f"{column}_liwc_tokens"].values:
        cat_counts = defaultdict(lambda:0)
        for token in v:
            for liwc_cat in token:
                cat_counts[liwc_cat] += 1

        for col in new_columns.keys():
            new_columns[col].append(cat_counts[col])

    for col in new_columns:
        df[col] = new_columns[col]

    df['num_tokens'] = [len(v) for v in df[f"{column}_tokens"]]
    """
    save data
    """
    os.makedirs(savedir, exist_ok=True)
    df.to_pickle(os.path.join(savedir, f"{savename}.fullframe.pickle"))
    print("full dataframe saved to:", os.path.join(savedir, f"{savename}.fullframe.pickle"))

    print("liwc keys:")
    print(new_col_keys)


def process_sentences(lsentences):
    """
    lsentences: list of strings
    returns lists with liwc tokens per each word token and the word tokens
    """
    liwc_tokens = []
    word_tokens = []
    
    for sent in tqdm(lsentences, 'Getting LIWC categories'):
        tokens = liwc.preprocess(sent)
        liwc_sentence = liwc.get_all_liwc_parts(tokens)

        liwc_tokens.append(liwc_sentence)
        word_tokens.append(tokens)

    return liwc_tokens, word_tokens

if __name__ == '__main__':
    main()