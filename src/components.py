
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

        return self

    @validator('examples')
    def min_len_examples(cls, v):
        if not len(v) >= 1:
            raise ValueError(
                f'Not Enough examples to compile, given: {len(v)}, expected at least 1'
            )
        return v[:10]

class WordSimilarities(BaseModel):
    word:str
    year:int
    props:List[float]

class WordFitted(BaseModel):
    word:str
    sense:str
    years:List[int]
    props:List[float]
    poly_fit:List[float]


class SenseEmbedding(BaseModel):
    id: str = None
    definition: str = None
    embedding: List[float] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.id = kwargs['id']
        self.definition = kwargs['definition']
        self.embedding = kwargs['embedding']

        return self


class Embedding(BaseModel):
    word: str = None
    sentence_number_index: List[List] = None
    embeddings: List[List] = None

    class Config:
        allow_population_by_field_name = True

    def __call__(self, **kwargs):
        self.word = kwargs['word']
        self.sentence_number_index = kwargs['sentence_number_index']
        self.embeddings = kwargs['embeddings']

        return self