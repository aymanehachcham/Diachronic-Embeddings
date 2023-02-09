
import json
import os
from src.api_call import OxfordDictAPI


def create_sens_embeddings(path:str):
    if os.path.exists(path):
        with open(path) as f: full_text = f.read()
        all_words = []
        for word in full_text.split('\n'):
            try:
                out_dict = OxfordDictAPI(word_id=word).get_senses()
                all_words.append(out_dict)
            except ValueError:
                continue
        return all_words
    else:
        raise ValueError(
            'Path given does not exist: {}'.format(path)
        )

def perform_on_all_words(path:str):
    if not os.path.exists(path):
        raise ValueError(
            'Path given does not exist: {}'.format(path)
        )
    with open(path) as f: full_text = f.read()
    all_words = []
    for word in full_text.split('\n'):
        print(f'Extracting examples for the word {word}. {"-"*10}')
        try:
            out_dict = OxfordDictAPI(word_id=word).get_senses()
            all_words.append(out_dict)
        except ValueError:
            continue
    return all_words



def poly_words(list_words, num_senses: int):
    for word in list_words:
        if len(word) > num_senses:
            yield (word[0]['word'], word)


if __name__ == '__main__':
    with open('../data/target_words/senses_oxford_api.json', 'w') as f:
        json.dump(perform_on_all_words('../data/target_words/polysemous.txt'), f, indent=4)