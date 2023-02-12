
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

    def polynomial_fit(self, sense: int, deg:int):
        years = [1980, 1982, 1985, 1987, 1989, 1990, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012,
                     2013, 2015, 2016, 2017, 2018]
        dist = self.sense_distribution(sense)

        return np.poly1d(np.polyfit(years, dist, deg)), years

def plot_word(word:str):

    poly_w1 = PolynomialFitting(word=word)

    xp = np.linspace(1980, 2018, 100)
    dist_1 = poly_w1.sense_distribution(sense_idx=sense_id_w1)

    p_1, y_1 = poly_w1.polynomial_fit(sense=sense_id_w1, deg=20)
    p_2, _ = poly_w2.polynomial_fit(sense=sense_id_w2, deg=20)

    fig, ax = plt.subplots()
    ax.plot(y_1, dist_1, '*', xp, p_1(xp), '-', label=f'{w_1}, for the sense: {sense_id_w1}')
    ax.plot(y_1, dist_2, '*', xp, p_2(xp), '-', label=f'{w_2} for the sense: {sense_id_w2}')

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

    plt.ylim(0, 1)
    ax.legend()
    plt.show()


if __name__ == '__main__':
    import json
    import numpy as np
    import matplotlib.pyplot as plt

    plot_words(('kill', 'black'), sense_id_w1=0, sense_id_w2=1)


    # poly = PolynomialFitting(word='abuse')
    # # print(p.sense_distribution(4))
    #
    # xp = np.linspace(1980, 2018, 100)
    # dist = poly.sense_distribution(sense_idx=0)
    # p, y = poly.polynomial_fit(sense=0, deg=20)
    # plt.plot(y, dist, '*', xp, p(xp), '-')
    # plt.ylim(0, 1)
    # plt.show()

    # word = 'abuse'
    # with open(f'../embeddings_similarity/embeddings_sim_{word}.json', 'r') as f:
    #     sims = json.load(f)
    #
    # sense_1 = []
    # sense_2 = []
    # sense_3 = []
    # sense_4 = []
    # sense_5 = []
    # sense_6 = []
    # years = []
    # for sim in sims:
    #     y = sim['year']
    #     s1 = sim['props'][0]
    #     s2 = sim['props'][1]
    #     s3 = sim['props'][2]
    #     s4 = sim['props'][3]
    #     s5 = sim['props'][4]
    #     s6 = sim['props'][5]
    #     # s3 = sim['props'][2]
    #     sense_1 += [s1]
    #     sense_2 += [s2]
    #     sense_3 += [s3]
    #     sense_4 += [s4]
    #     sense_5 += [s5]
    #     sense_6 += [s6]
    #     # sense_3 += [s2]
    #     years += [y]
    #
    # xp = np.linspace(1980, 2018, 100)
    # p = np.poly1d(np.polyfit(years, sense_1, 20))
    # p2 = np.poly1d(np.polyfit(years, sense_2, 20))
    # p3 = np.poly1d(np.polyfit(years, sense_3, 20))
    # p4 = np.poly1d(np.polyfit(years, sense_4, 20))
    # p5 = np.poly1d(np.polyfit(years, sense_5, 20))
    # p6 = np.poly1d(np.polyfit(years, sense_6, 20))
    # # p3 = np.poly1d(np.polyfit(years, sense_3, 20))
    # import matplotlib.pyplot as plt
    #
    # plt.plot(years, sense_1, '.', years, sense_2, '*', years, sense_3, '+', years, sense_4, '-', years, sense_5, '--', years, sense_6, 'o',
    #          xp, p(xp), '-', xp, p2(xp), '--', xp, p3(xp), '--', xp, p4(xp), '--', xp, p5(xp), '--', xp, p6(xp), '--')
    # plt.ylim(0, 1)
    # plt.show()



    # sense_1 = []
    # sense_2 = []
    # sense_3 = []
    # sense_4 = []
    # sense_5 = []
    # sense_6 = []
    # years = []
    # for year in range(1980, 2020, 5):
    #     with open(f'../embeddings_similarity/embeddings_sim_{year}.json', 'r') as f:
    #         sims = json.load(f)
    #
    #     abuse = Word(**sims[0])
    #     proportion_sense_0 = abuse.props[0]
    #     proportion_sense_1 = abuse.props[1]
    #     proportion_sense_2 = abuse.props[2]
    #
    #     sense_1.append(proportion_sense_0)
    #     sense_2.append(proportion_sense_1)
    #     sense_3.append(proportion_sense_2)
    #
    #     years.append(year)
    #
    # xp = np.linspace(1980, 2015, 100)
    # p = np.poly1d(np.polyfit(years, sense_1, 20))
    # p2 = np.poly1d(np.polyfit(years, sense_2, 20))
    # p3 = np.poly1d(np.polyfit(years, sense_3, 20))
    # print(sense_1, '\n', sense_2)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(years, sense_1, '.', years, sense_2, '*',
    #          years, sense_3, '+',
    #          xp, p(xp), '-', xp, p2(xp), '--', xp, p3(xp), '--')
    # plt.ylim(0, 1)
    # plt.show()





    # print(np.polyfit(words_80[0]['props'], words_85[0]['props'], deg=3))
    # print(PolynomialFitting(root_dir='../data/target_words/senses_proportions_1980.json').fit_polynomila())