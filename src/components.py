
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import numpy as np

class OxfordAPIResponse(BaseModel):
    id:str = None
    definition:str = None
    examples:Optional[List[str]] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.id = kwargs['id']
        self.definition = kwargs['definition']
        self.examples = kwargs['examples']

    @validator('examples')
    def min_len_examples(cls, v):
        if not len(v) >= 5:
            raise ValueError(
                f'Not Enough examples to compile, given: {len(v)}, expected at least 5'
            )
        return v

class SenseEmbedding(BaseModel):
    sense:str = None
    definition:str = None
    embedding:List[float] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.id = kwargs['id']
        self.definition = kwargs['definition']
        self.examples = kwargs['embedding']

    @property
    def get_embeddings(self):
        if self.embedding is None:
            raise ValueError(
                f'The Embeddings provided are null: {self.embedding}'
            )
        return np.array(self.embedding)


class Word(BaseModel):
    word:str
    props:List[float]

class WordFitted(BaseModel):
    word:str
    sense:str
    years:List[int]
    props:List[float]
    poly_fit:List[float]




class Embedding(BaseModel):
    word: str
    sentence_number_index: List[List]
    embeddings: List[List]

    @property
    def get_embeddings(self):
        if self.embeddings is None:
            raise ValueError(
                f'The Embeddings provided are null: {self.embeddings}'
            )
        return np.array(self.embeddings)


if __name__ == '__main__':
    w = Word(word='lola', props=[0.4, 0.5])
    w.email = 'hello'