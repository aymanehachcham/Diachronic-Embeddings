
import os.path
from typing import List
from components import WordSimilarities
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class PolynomialFitting:
    word:str
    years:List[str] = None
    def _load_elements(self):
        with open(f'../embeddings_similarity/embeddings_sim_{self.word}.json') as f:
            word_props = json.load(f)

        sense_proportion_distribution = []
        for w_ in word_props:
             sense_proportion_distribution += [WordSimilarities(**w_).props]

        return sense_proportion_distribution

    def sense_distribution(self, sense_idx):
        proportions = self._load_elements()
        if not sense_idx in range(0, len(proportions[0])+1):
            raise ValueError(
                f'The sense index {sense_idx} not present in the range of senses available for the word {self.word}'
            )
        return [props_[sense_idx] for props_ in proportions]

    def polynomial_fit(self, sense: int, deg:int=20):
        years = [1980, 1982, 1985, 1987, 1989, 1990, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012,
                     2013, 2015, 2016, 2017, 2018]
        dist = self.sense_distribution(sense)

        return np.poly1d(np.polyfit(years, dist, deg)), years

    def distribution_all_senses(self, senses:list):
        all_senses = []
        sense_ = {}
        for sense_num in senses:
            fit, y = self.polynomial_fit(sense_num)
            sense_['years'] = y
            sense_['distribution'] = self.sense_distribution(sense_num)
            sense_['polynomial_fit'] = fit
            sense_['sense_id'] = sense_num

            all_senses += [sense_.copy()]

        return all_senses

def plot_word(word:str):
    poly_w1 = PolynomialFitting(word=word)
    xp = np.linspace(1980, 2018, 100)

    fig, ax = plt.subplots()
    objects = ['+', '*', '-', 'o']

    for sense, obj in zip(poly_w1.distribution_all_senses([0, 1, 2, 3]), objects):
        ax.plot(sense['years'], sense['distribution'], f'{obj}', xp, sense['polynomial_fit'](xp), '-', label=f'{word}, for the sense: {sense["sense_id"]}')

    plt.ylim(0, 1)
    ax.legend()
    plt.show()

def plot_words(words:tuple, sense_id_w1:int, sense_id_w2:int):
    w_1, w_2 = words

    poly_w1 = PolynomialFitting(word=w_1)
    poly_w2 = PolynomialFitting(word=w_2)

    xp = np.linspace(1980, 2018, 100)
    dist_1 = poly_w1.sense_distribution(sense_idx=sense_id_w1)
    dist_2 = poly_w2.sense_distribution(sense_idx=sense_id_w2)

    p_1, y_1 = poly_w1.polynomial_fit(sense=sense_id_w1, deg=20)
    p_2, _ = poly_w2.polynomial_fit(sense=sense_id_w2, deg=20)

    fig, ax = plt.subplots()
    ax.plot(y_1, dist_1, '*', xp, p_1(xp), '-', label=f'{w_1}, for the sense: {sense_id_w1}')
    ax.plot(y_1, dist_2, '*', xp, p_2(xp), '-', label=f'{w_2} for the sense: {sense_id_w2}')

    # plt.ylim(0, 0.2)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    plot_words(('abuse', 'black'), sense_id_w1=2, sense_id_w2=2)