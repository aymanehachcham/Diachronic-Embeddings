
import os
from api_call import OxfordDictAPI

def perform_on_all_words(path:str):
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


def poly_words(list_words, num_senses: int):
    for word in list_words:
        if len(word) > num_senses:
            yield (word[0]['word'], word)