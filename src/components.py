
from pydantic import BaseModel, validator, Field
from typing import List, Optional
import numpy as np

class Word(BaseModel):
    word:str
    props:List[float]

class WordFitted(BaseModel):
    word:str
    sense:str
    years:List[int]
    props:List[float]
    poly_fit:List[float]

class OxfordAPIResponse(BaseModel):
    id:str
    definition:str
    examples:Optional[List[str]] = Field(None, alias="examples")

    @validator('examples')
    def min_len_examples(cls, v):
        if not len(v) > 10:
            raise ValueError(
                f'Not Enough examples to compile, given: {len(v)}, expected at least 10'
            )
        return v

    class Config:
        allow_population_by_field_name = True


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


if __name__ == '__main__':
    w = Word(word='lola', props=[0.4, 0.5])
    w.email = 'hello'