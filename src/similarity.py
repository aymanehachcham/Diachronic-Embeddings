
import os
import logging
import json
from numpy.linalg import norm
from collections import Counter

import numpy as np

SENSES_FILE = 'embeddings_for_senses.json'

class LoadingEmbeddings():

    @classmethod
    def load_files(
            cls,
            root_dir:str,
            sense_embeddings_file: str,
            example_embeddings_file: str,
        ):

        logging.basicConfig(level=logging.NOTSET)
        example_embeds = []
        with open(os.path.join(root_dir, example_embeddings_file), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-' * 10, f.name, '-' * 10))
            for line in f:
                example_embeds.append(json.loads(line))
        # with open(os.path.join(root_dir, example_embeddings_file), 'r') as f:
        #     logging.info('{} Loading File {} {}'.format('-' * 10, f.name, '-' * 10))
        #     example_embeds = json.load(f)

        with open(os.path.join(root_dir, sense_embeddings_file), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-' * 10, f.name, '-' * 10))
            senses_embeds = json.load(f)

        return example_embeds, senses_embeds


class Similarities():
    def __init__(
            self,
            senses_file:str,
            examples_file: str
        ):

        self.embeddings_examples, self.embeddings_senses = LoadingEmbeddings.load_files(
            root_dir='../embeddings',
            sense_embeddings_file=senses_file,
            example_embeddings_file=examples_file
        )
        self.word_sense_proportions = {}

    def _search_word_sense(self, word:str):
        for w in self.embeddings_senses:
            if w[0]['word'] == word:
                yield w
            else:
                continue

    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, word:str):
        from collections import Counter
        # examples = [np.array(ex['embeddings']) for ex in self.embeddings_examples]
        # examples = np.array(self.embeddings_examples[word]['embeddings'])
        try:
            senses = next(self._search_word_sense(word))
        except StopIteration:
            raise ValueError(
                'Word not in list'
            )

        all_sims = []
        for word in self.embeddings_examples:
            print(f'{"-"*10} Similarities for the word {word["word"]} {"-"*10}')
            for embed in word['embeddings']:
                s_argmax =  np.argmax(list(self._cos_sim(np.array(sens['embedding']), np.array(embed)) for sens in senses))
                all_sims.append(senses[s_argmax]['sense'])

        self.word_sense_proportions['word'] = word
        self.word_sense_proportions['sense_distribution'] = all_sims
        self.word_sense_proportions['props'] = list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))

        return self.word_sense_proportions.copy()


def sim_on_all_words():
    with open('../data/target_words/polysemous.txt', 'r') as f:
        words = f.read()

    for year in range(1980, 2020, 5):
        print(f'------------ {year} ---------------')
        sim = Similarities(
            senses_file=SENSES_FILE,
            examples_file=f'embeddings_{year}.json'
        )
        with open(f'../embeddings_similarity/embeddings_sim_{year}.json', 'w') as f:
            json.dump([sim(w) for w in words.split('\n')], f, indent=4)


if __name__ == '__main__':
    # d = []
    # with open('../embeddings/embeddings_' + '1980' + '.json', 'r') as f:
    #     for line in f:
    #         d.append(json.loads(line))
    #
    # print(d[0][''])
    # with open('../data/target_words/polysemous.txt', 'r') as f:
    #     # words = f.read()
    sim_on_all_words()
    # import json
    #
    # sim = Similarities()
    # all_words = []
    #
    # with open('../data/target_words/polysemous.txt', 'r') as f:
    #     words = f.read()
    #
    # for word in words.split('\n'):
    #     all_words.append(sim(word))
    #
    # with open('../embeddings/embeddings_sim_1980.json', 'w') as f:
    #     json.dump(all_words, f, indent=4)