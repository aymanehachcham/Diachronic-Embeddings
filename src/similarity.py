
import os
import json
from numpy.linalg import norm
from components import Embedding, SenseEmbedding
from settings import EmbeddingFiles, FileLoader
import logging
import numpy as np

class Similarities():
    def __init__(
            self,
        ):

        self.files = EmbeddingFiles()
        self.embedding_component = Embedding()
        self.sense_component = SenseEmbedding()
        self.embeddings_senses, self.words = FileLoader.load_files(self.__class__.__name__)

        self.embeddings_examples = None
        self.root_dir = self.files.embeddings_root_dit
        self.word_sense_proportions = {}


    def _lookup_examples_that_match(self, words):
        if self.embeddings_examples is None:
            raise ValueError(
                'Embedding examples not initialized'
            )

        embeddings = []
        for embed in self.embeddings_examples:
            embeddings += [embed['word']]
        for word in list(set(embeddings) & set(words.split('\n'))):
            yield next(self.embedding_component(**e) for e in self.embeddings_examples if word == e['word'])

    def _search_word_sense(self, main_word:str):
        for w in self.embeddings_senses:
            if not w['word'] == main_word:
                continue
            return [self.sense_component(**s) for s in w['senses']]


    def _cos_sim(self, vect_a:np.array, vect_b:np.array):
        return (vect_a @ vect_b)/(norm(vect_a) * norm(vect_b))

    def __call__(self, main_word:str, year:int, path_embeddings_file:str):
        from collections import Counter
        print(f'{"-" * 10} Computing Similarities for the word {main_word} {"-" * 10}')

        with open(os.path.join(self.root_dir, path_embeddings_file), 'r') as f:
            logging.info(f'{"-" * 10} Loading the embeddings examples file: {f.name} {"-" * 10}')
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

            self.word_sense_proportions['word'] = main_word
            self.word_sense_proportions['year'] = year
            self.word_sense_proportions['props'] = list(map(lambda x: x[1]/len(all_sims), Counter(all_sims).most_common()))

        return self.word_sense_proportions.copy()


def sim_on_all_words():
    sim = Similarities()

    with open(sim.files.poly_words_f, 'r') as f:
        words = f.read()

    for w_ in words.split('\n')[:1]:
        s = []
        for year in [1980]:

            s += [sim(w_, year, path_embeddings_file=f'embeddings_{year}.json')]

        with open(f'../embeddings_similarity/embeddings_sim_{"w_"}.json', 'w') as f:
            json.dump(s, f, indent=4)
        break


if __name__ == '__main__':
    sim_on_all_words()


