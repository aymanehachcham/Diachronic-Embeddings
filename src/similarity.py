
import os
import logging
import json
from numpy.linalg import norm
from collections import Counter

import numpy as np

SENSES_FILE = 'embeddings_for_senses.json'
EXAMPLES_FILE = 'embeddings_2005.json'
WORDS = 'polysemous.txt'

class LoadingEmbeddings():

    @classmethod
    def load_files(
            cls,
            root_dir:str,
            sense_embeddings_file: str,
            example_embeddings_file: str,
        ):

        if (os.path.exists(os.path.join(root_dir, sense_embeddings_file))) \
            and (os.path.exists(os.path.join(root_dir, sense_embeddings_file))):
            cls.sense_file = os.path.join(root_dir, sense_embeddings_file)
            cls.examples_file = os.path.join(root_dir, example_embeddings_file)

        logging.basicConfig(level=logging.NOTSET)
        ex, sens = cls._load_files()
        return ex, sens

    @staticmethod
    def _load_files():
        with open(os.path.join('../embeddings', EXAMPLES_FILE), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-'*10, f.name, '-'*10))
            example_embeds = json.load(f)

        with open(os.path.join('../embeddings', SENSES_FILE), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-'*10, f.name, '-'*10))
            senses_embeds = json.load(f)

        return example_embeds, senses_embeds


class Similarities():
    def __init__(
            self,
        ):

        self.embeddings_examples, self.embeddings_senses = LoadingEmbeddings.load_files(
            root_dir='../embeddings',
            sense_embeddings_file=SENSES_FILE,
            example_embeddings_file=EXAMPLES_FILE
        )
        self.word_sense_proportions = {}

    def _search_word_sense(self, word:str):
        for w in self.embeddings_senses:
            if w[0]['word'] == word:
                yield w

    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, word:str):
        from collections import Counter
        examples = np.array(self.embeddings_examples[word]['embeddings'])
        try:
            senses = next(self._search_word_sense(word))
        except StopIteration:
            raise ValueError(
                'Word not in list'
            )

        all_sims = []
        for embed in examples:
            s_argmax =  np.argmax(list(self._cos_sim(sens['embedding'], embed) for sens in senses))
            all_sims.append(senses[s_argmax]['sense'])

        self.word_sense_proportions['word'] = word
        self.word_sense_proportions['props'] = list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))

        return self.word_sense_proportions.copy()



if __name__ == '__main__':
    import json

    sim = Similarities()
    all_words = []

    with open('../data/target_words/polysemous.txt', 'r') as f:
        words = f.read()

    for word in words.split('\n'):
        all_words.append(sim(word))

    with open('../data/target_words/senses_proportions_2005.json', 'w') as f:
        json.dump(all_words, f, indent=4)