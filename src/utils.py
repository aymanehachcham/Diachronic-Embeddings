
import json
import os
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
    if not os.path.exists(path):
        raise ValueError(
            f'No matches for the given path: {path}'
        )
    with open(path) as f: full_text = f.read()
    for word in full_text.split('\n'):
        try:
            all_words += [OxfordDictAPI(word_id=word).get_senses()]
        except ValueError:
            continue

    return all_words

def poly_words(list_words, num_senses: int):
    for word in list_words:
        if len(word) > num_senses:
            yield (word[0]['word'], word)


if __name__ == '__main__':
    files = EmbeddingFiles()
    with open(files.oxford_word_senses, 'r') as f:
        json.dump(perform_on_all_words(files.poly_words_f), f, indent=4)

