
import os
import logging
import json
from numpy.linalg import norm
from pydantic import BaseModel, validator
from typing import List

import numpy as np

SENSES_FILE = 'embeddings_for_senses.json'

class SenseEmbedding(BaseModel):
    word:str
    sense:str
    definition:str
    embedding:List[float]

    @property
    def get_embeddings(self):
        if self.embedding is None:
            raise ValueError(
                f'The Embeddings provided are null: {self.embedding}'
            )
        return np.array(self.embedding)

class Embedding(BaseModel):
    sentence_number_index: List[List]
    embeddings: List[List]
    word:str
    @property
    def get_embeddings(self):
        if self.embeddings is None:
            raise ValueError(
                f'The Embeddings provided are null: {self.embeddings}'
            )
        return np.array(self.embeddings)

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
        with open('../data/target_words/polysemous.txt', 'r') as f:
            words = f.read()

        self.all_embeddings = []
        for embed, w in zip(self.embeddings_examples, words.split('\n')):
            v = Embedding(**embed)
            if v.word != w:
                v.word = w

            self.all_embeddings.append(v.copy())

    def _search_word_sense(self, main_word:str):
        for w in self.embeddings_senses:
            if w[0]['word'] == main_word:
                yield w

    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, main_word:str):
        from collections import Counter
        # examples = [np.array(ex['embeddings']) for ex in self.embeddings_examples]
        # examples = np.array(self.embeddings_examples[word]['embeddings'])
        try:
            senses = next(self._search_word_sense(main_word))
        except StopIteration:
            raise ValueError(
                'Word not in list'
            )

        all_sims = []
        for embed in self.all_embeddings:
            if embed.word == main_word:
                print(f'{"-"*10} Similarities for the word {embed.word} {"-"*10}')
                for embedding_ex in embed.embeddings:
                    s_argmax =  np.argmax(list(self._cos_sim(np.array(sens['embedding']), np.array(embedding_ex)) for sens in senses))
                    all_sims.append(senses[s_argmax]['sense'])
        #
            self.word_sense_proportions['word'] = main_word
        # self.word_sense_proportions['sense_distribution'] = all_sims
            self.word_sense_proportions['props'] = list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))
        #
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
        similarities = []
        with open(f'../embeddings_similarity/embeddings_sim_{year}.json', 'w') as f:
            for w in words.split('\n'):
                try:
                    similarities.append(sim(w))
                except ValueError:
                    continue

            json.dump(similarities, f, indent=4)


if __name__ == '__main__':
    sim_on_all_words()

    # with open('../data/target_words/polysemous.txt', 'r') as f:
    #     words = f.read()
    #
    # sim = Similarities(
    #     senses_file=SENSES_FILE,
    #     examples_file=f'embeddings_{1980}.json'
    # )
    #
    # sims = []
    # for w in words.split('\n'):
    #     sims.append(sim(w))
    #
    # with open(f'../embeddings_similarity/embeddings_sim_{1980}.json', 'w') as f:
    #     json.dump(sims, f, indent=4)


    # embeddings_examples, embeddings_senses = LoadingEmbeddings.load_files(
    #     root_dir='../embeddings',
    #     sense_embeddings_file=SENSES_FILE,
    #     example_embeddings_file='embeddings_1980.json'
    # )
    #
    # with open('../data/target_words/polysemous.txt', 'r') as f:
    #     words = f.read()
    #
    # all_embeds = []
    # for embed, w in zip(embeddings_examples, words.split('\n')):
    #     v = Embedding(**embed)
    #     if Embedding(**embed).word != w:
    #         v.word = w
    #
    #     all_embeds.append(v.copy())
    #     if Embedding(**embed).word != w:
    #         print(w)

