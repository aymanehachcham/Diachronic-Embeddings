
import os.path
from pydantic import BaseModel, validator
from typing import Union, List

class Word(BaseModel):
    word:str
    props:List[float]


class PolynomialFitting(BaseModel):
    root_dir:str
    words_80:List[Word]
    words_85: List[Word]
    @validator('root_dir')
    def check_path(cls, value):
        if not os.path.exists(value):
            raise ValueError(
                f'The Path given: {value} is not valid'
            )
        return value

    # Load files should be a Config file, research on that:
    def _load_files(self):
        with open('../data/target_words/senses_proportions_1980.json') as f:
            self.words_80 = json.load(f)

        with open('../data/target_words/senses_proportions_1985.json') as f:
            self.words_85 = json.load(f)

    def fit_polynomila(self):
        return [np.polyfit(Word(**w1).props, Word(**w2).props) for w1, w2 in zip(self.words_80, self.words_85)]



if __name__ == '__main__':
    import json
    import numpy as np

    with open('../data/target_words/senses_proportions_1980.json') as f:
        words_80 = json.load(f)

    with open('../data/target_words/senses_proportions_1985.json') as f:
        words_85 = json.load(f)

    # print(np.polyfit(words_80[0]['props'], words_85[0]['props'], deg=3))
    print(PolynomialFitting(root_dir='../data/target_words/senses_proportions_1980.json').fit_polynomila())