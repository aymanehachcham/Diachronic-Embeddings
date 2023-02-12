
import json
import logging
import os
import time

from src.api_call import OxfordDictAPI
from settings import EmbeddingFiles

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
    all_words = []
    processed_words = []
    if not os.path.exists(path):
        raise ValueError(
            f'No matches for the given path: {path}'
        )
    with open(path) as f: full_text = f.read()
    for idx, word in enumerate(full_text.split('\n')):
        try:
            if idx % 10 == 0:
                print('sleeping')
                time.sleep(3)
            all_words += [OxfordDictAPI(word_id=word).get_senses()]
            processed_words += [word]
        except ValueError:
            continue

    return all_words, processed_words

def poly_words(list_words, num_senses: int):
    for word in list_words:
        if len(word) > num_senses:
            yield (word[0]['word'], word)


if __name__ == '__main__':
    files = EmbeddingFiles()

    with open(files.oxford_word_senses, 'w') as f:
        w, p = perform_on_all_words(files.poly_words_f)
        json.dump(w, f, indent=4)
        print(p)

