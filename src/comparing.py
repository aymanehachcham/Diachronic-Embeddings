
import json

import numpy as np

if __name__ == '__main__':
    with open('../data/target_words/senses_proportions_1980.json', 'r') as f:
        words_80 = json.load(f)

    with open('../data/target_words/senses_proportions_1980.json', 'r') as f:
        words_85 = json.load(f)

    with open('../data/target_words/senses_proportions_1980.json', 'r') as f:
        words_90 = json.load(f)


    print(np.array(words_85[0]['props']))