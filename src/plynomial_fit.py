
import random
from components import WordSimilarities
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import BSpline, splrep
from typing import Literal, List
import warnings
from settings import EmbeddingFiles


class PolynomialFitting:
    def __init__(
            self,
            word:str
        ):
        self.files = EmbeddingFiles()
        self.sense_proportion_distribution = []
        self.word = word
        self.years = self.files.years_used

        with open(f'../embeddings_similarity/embeddings_sim_{self.word}.json') as f:
            word_props = json.load(f)

        with open(self.files.poly_words_f, 'r') as f:
            self.words = f

        for w_ in word_props:
             self.sense_proportion_distribution += [WordSimilarities(**w_).props]
        self.num_senses = len(self.sense_proportion_distribution[0])


    def sense_distribution(self, sense_idx):
        if not sense_idx in range(0, self.num_senses + 1):
            raise ValueError(
                f'The sense index {sense_idx} not present in the range of senses available for the word {self.word}'
            )
        return [props_[sense_idx] for props_ in self.sense_proportion_distribution]

    def polynomial_fit(self, sense: int, deg:int=20):
        warnings.filterwarnings('ignore')
        dist = self.sense_distribution(sense)
        return np.poly1d(np.polyfit(self.years, dist, deg))

    def spline_fit(self, sense:int):
        dist = self.sense_distribution(sense)
        tck_spline_args = splrep(self.years, dist, s=0, k=3)
        return BSpline(*tck_spline_args)

    def distribution_all_senses(self, fit:Literal['polynomial', 'bspline']):
        all_senses = []
        sense_ = {}
        xp = np.linspace(1980, 2018, 100)
        for sense_num in range(0, self.num_senses):
            sense_['years'] = self.years
            sense_['distribution'] = self.sense_distribution(sense_num)
            sense_['sense_id'] = sense_num

            if fit == 'polynomial':
                sense_['y_fit'] = self.polynomial_fit(sense_num)

            if fit == 'bspline':
                sense_['y_fit'] = self.spline_fit(sense_num)(xp)

            all_senses += [sense_.copy()]

        return all_senses

def plot_word(word:str, fit:Literal['polynomial', 'bspline']):
    if not fit in ['polynomial', 'bspline']:
        raise ValueError(
            f'The fit type provided is not correct, expected "polynomial" or "bspline", got {type(fit)}'
        )

    fig, ax = plt.subplots()
    markers = ['o', 'v', '^', 's', 'p', 'P', 'h', 'H', 'D']
    random.shuffle(markers)
    poly_w1 = PolynomialFitting(word=word)
    xp = np.linspace(1980, 2018, 100)

    if fit == 'polynomial':
        distr_all_senses = poly_w1.distribution_all_senses(fit='polynomial')
        for sense, obj in zip(distr_all_senses, markers[:poly_w1.num_senses]):
            ax.plot(sense['years'], sense['distribution'], f'{obj}', label=f'{word}, for the sense: {sense["sense_id"]}')
            ax.plot(xp, sense['y_fit'](xp), '-')

    if fit == 'bspline':
        distr_all_senses = poly_w1.distribution_all_senses(fit='bspline')
        for sense, obj in zip(distr_all_senses, markers[:poly_w1.num_senses]):
            ax.plot(sense['years'], sense['distribution'], f'{obj}', label=f'{word}, for the sense: {sense["sense_id"]}')
            ax.plot(xp, sense['y_fit'], '-')

    plt.ylim(0, 1)
    ax.legend()
    plt.show()

def plot_words(words:tuple, sense_id_w1:int, sense_id_w2:int, sense_id_w3:int, fit:Literal['polynomial', 'bspline']):
    w_1, w_2, w_3= words

    poly_w1 = PolynomialFitting(word=w_1)
    poly_w2 = PolynomialFitting(word=w_2)
    poly_w3 = PolynomialFitting(word=w_3)

    xp = np.linspace(1980, 2018, 100)
    fig, ax = plt.subplots()
    dist_1 = poly_w1.sense_distribution(sense_idx=sense_id_w1)
    dist_2 = poly_w2.sense_distribution(sense_idx=sense_id_w2)
    dist_3 = poly_w3.sense_distribution(sense_idx=sense_id_w3)

    if fit == 'polynomial':
        p_1 = poly_w1.polynomial_fit(sense=sense_id_w1, deg=20)
        p_2 = poly_w2.polynomial_fit(sense=sense_id_w2, deg=20)
        p_3 = poly_w3.polynomial_fit(sense=sense_id_w3, deg=20)

        ax.plot(poly_w1.years, dist_1, '*', label=f'{w_1}, for the sense: {sense_id_w1}')
        ax.plot(xp, p_1(xp), '-', )
        ax.plot(poly_w1.years, dist_2, '*', label=f'{w_2} for the sense: {sense_id_w2}')
        ax.plot(xp, p_2(xp), '-',)

        ax.plot(poly_w1.years, dist_3, '+', label=f'{w_3} for the sense: {sense_id_w3}')
        ax.plot(xp, p_3(xp), '-', )

    if fit == 'bspline':
        b_1 = poly_w1.spline_fit(sense=sense_id_w1)
        b_2 = poly_w2.spline_fit(sense=sense_id_w2)
        b_3 = poly_w3.spline_fit(sense=sense_id_w3)

        ax.plot(poly_w1.years, dist_1, '*', label=f'{w_1}, for the sense: {sense_id_w1}')
        ax.plot(xp, b_1(xp), '-', )
        ax.plot(poly_w1.years, dist_2, '*', label=f'{w_2} for the sense: {sense_id_w2}')
        ax.plot(xp, b_2(xp), '-', )

        ax.plot(poly_w1.years, dist_3, '+', label=f'{w_3} for the sense: {sense_id_w3}')
        ax.plot(xp, b_3(xp), '-', )

    # plt.ylim(0, 0.2)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    plot_words(('abuse', 'black', 'kill'), sense_id_w1=2, sense_id_w2=2, sense_id_w3=2, fit='bspline')
    # plot_word('abuse', fit='bspline')

    # p = PolynomialFitting('abuse')
    # from scipy import interpolate
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # y = [0, 1, 3, 4, 3, 5, 7, 5, 2, 3, 4, 8, 9, 8, 7]
    # n = len(y)
    # x = range(0, n)
    #
    # tck = interpolate.splrep(x, y, s=0, k=3)
    # # x_new = np.linspace(min(x), max(x), 100)
