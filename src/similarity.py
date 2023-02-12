
import os
import logging
import json
from numpy.linalg import norm
from pydantic import BaseModel, validator
from typing import List
import logging

import numpy as np

SENSES_FILE = 'embeddings_for_senses.json'

class SenseEmbedding(BaseModel):
    id:str
    definition:str
    embedding:List[float]


class Embedding(BaseModel):
    word: str
    sentence_number_index: List[List]
    embeddings: List[List]

class LoadingEmbeddings():

    @classmethod
    def load_files(
            cls,
            root_dir:str,
            sense_embeddings_file: str
        ):

        logging.basicConfig(level=logging.NOTSET)

        with open(os.path.join(root_dir, sense_embeddings_file), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-' * 10, f.name, '-' * 10))
            senses_embeds = json.load(f)

        return senses_embeds


class Similarities():
    def __init__(
            self,
            senses_file:str,
        ):

        self.embeddings_senses = LoadingEmbeddings.load_files(
            root_dir='../embeddings',
            sense_embeddings_file=senses_file,
        )

        self.embeddings_examples = None
        self.root_dir = '../embeddings'
        self.word_sense_proportions = {}

        with open('../data/target_words/polysemous.txt', 'r') as f:
            self.words = f.read()


    def _lookup_examples_that_match(self, words):
        if self.embeddings_examples is None:
            raise ValueError(
                'Embedding examples not initalized'
            )

        embeddings = []
        for embed in self.embeddings_examples:
            embeddings += [embed['word']]
        for word in list(set(embeddings) & set(words.split('\n'))):
            yield next(Embedding(**e) for e in self.embeddings_examples if word == e['word'])

    def _search_word_sense(self, main_word:str):
        for w in self.embeddings_senses:
            if not w['word'] == main_word:
                continue
            return [SenseEmbedding(**s) for s in w['senses']]


    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, main_word:str, year:int, path_embeddings_file:str):
        from collections import Counter
        print(f'{"-" * 10} Similarities for the word {main_word} {"-" * 10}')

        with open(os.path.join(self.root_dir, path_embeddings_file), 'r') as f:
            logging.info('{} Loading File {} {}'.format('-' * 10, f.name, '-' * 10))
            self.embeddings_examples = json.load(f)

        self.all_embeddings = list(self._lookup_examples_that_match(self.words))

        try:
            w_senses = self._search_word_sense(main_word)
        except StopIteration:
            raise ValueError(
                f'Word {main_word} not present in the list of words'
            )

        all_sims = []
        for embed in self.all_embeddings:
            for ex in embed.embeddings:
                s_argmax = np.argmax([self._cos_sim(np.array(ex), np.array(s.embedding)) for s in w_senses])
                all_sims += [w_senses[s_argmax].id]
        #
            self.word_sense_proportions['word'] = main_word
            self.word_sense_proportions['year'] = year
            self.word_sense_proportions['props'] = list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))
        #
        return self.word_sense_proportions.copy()


def sim_on_all_words():
    with open('../data/target_words/polysemous.txt', 'r') as f:
        words = f.read()

    sim = Similarities(senses_file=SENSES_FILE)

    for w_ in words.split('\n')[1:]:
        s = []
        for year in [1980, 1982, 1985, 1987, 1989, 1990, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012,
                     2013, 2015, 2016, 2017, 2018]:

            s += [sim(w_, year, path_embeddings_file=f'embeddings_{year}.json')]

        with open(f'../embeddings_similarity/embeddings_sim_{w_}.json', 'w') as f:
            json.dump(s, f, indent=4)


if __name__ == '__main__':
    sim_on_all_words()


