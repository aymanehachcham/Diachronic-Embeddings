
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv(verbose=True)


class OxfordAPISettings(BaseSettings):
    accept:str = Field(..., env="ACCEPT")
    app_id:str = Field(..., env="APP_ID")
    app_key:str = Field(..., env="APP_KEY")
    url:str = Field(..., env="URL")
    strict_match:str = Field(..., env="STRICT_MATCH")

    def __str__(self):
        return self.accept + '\n' + self.app_id + '\n' + self.app_key + '\n' + self.url + '\n'

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"


class EmbeddingFiles(BaseSettings):
    poly_words_f:str = '../data/target_words/polysemous.txt'
    oxford_word_senses: str = '../data/target_words/senses_oxford_api.json'
    sense_embeddings: str = '../embeddings/embeddings_for_senses.json'
    embeddings_root_dit:str = '../embeddings'
    years_used = [1980, 1982, 1985, 1987, 1989, 1990, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012,
                     2013, 2015, 2016, 2017, 2018]

if __name__ == '__main__':
    print(OxfordAPISettings())

