
import os.path
from typing import List
from components import Word, WordFitted
from dataclasses import dataclass

@dataclass(frozen=True)
class PolynomialFitting:
    root_dir:str
    word:str

    def _load_files(self):
        for year in range(1980, 2020, 5):
            with open(f'../embeddings_similarity/embeddings_sim_{year}.json', 'r') as f:
                words = json.load(f)


if __name__ == '__main__':
    import json
    import numpy as np

    sense_1 = []
    sense_2 = []
    sense_3 = []
    sense_4 = []
    sense_5 = []
    sense_6 = []
    years = []
    for year in range(1980, 2020, 5):
        with open(f'../embeddings_similarity/embeddings_sim_{year}.json', 'r') as f:
            sims = json.load(f)

        abuse = Word(**sims[7])
        proportion_sense_0 = abuse.props[0]
        proportion_sense_1 = abuse.props[1]
        proportion_sense_2 = abuse.props[2]

        sense_1.append(proportion_sense_0)
        sense_2.append(proportion_sense_1)
        sense_3.append(proportion_sense_2)

        years.append(year)

    xp = np.linspace(1980, 2015, 100)
    p = np.poly1d(np.polyfit(years, sense_1, 20))
    p2 = np.poly1d(np.polyfit(years, sense_2, 20))
    p3 = np.poly1d(np.polyfit(years, sense_3, 20))
    print(sense_1, '\n', sense_2)

    import matplotlib.pyplot as plt

    plt.plot(years, sense_1, '.', years, sense_2, '*',
             years, sense_3, '+',
             xp, p(xp), '-', xp, p2(xp), '--', xp, p3(xp), '--')
    plt.ylim(0, 1)
    plt.show()





    # print(np.polyfit(words_80[0]['props'], words_85[0]['props'], deg=3))
    # print(PolynomialFitting(root_dir='../data/target_words/senses_proportions_1980.json').fit_polynomila())