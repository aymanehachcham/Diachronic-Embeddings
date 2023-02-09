import requests
from requests.models import Response
import json
from typing import List, Dict
from itertools import chain
import os
import re
from nltk import WordNetLemmatizer
from ..settings import OxfordAPISettings

API_CREDENTIALS = '../API_credentials/oxford_api_credentials.txt'
# def load_credentials(path:str):
#     if os.path.exists(path):
#         with open(path) as f: creds = f.read()
#         api_creds = {}
#         api_creds['Accept'] = creds.split('\n')[0].split(':')[1].strip()
#         api_creds['app_id'] = creds.split('\n')[1].split(':')[1].strip()
#         api_creds['app_key'] = creds.split('\n')[2].split(':')[1].strip()
#         api_creds['url'] = creds.split('\n')[3].split(':', 1)[1].strip()
#
#         return api_creds
#     else:
#         raise ValueError(
#             'Path given does not exist: {}'.format(path)
#         )

class OxfordDictAPI():
    def __init__(
            self,
            word_id: str
    ):
        if not isinstance(word_id, str):
            raise ValueError(
                f'Expected word_id to be a string, but got {type(word_id)}'
            )

        self.api_creds = OxfordAPISettings()
        self.word = word_id
        self.query = ('entries', 'sentences')
        self.url_ent = f'{self.api_creds.url}/{self.query[0]}/en/'
        self.url_sent = f'{self.api_creds.url}/{self.query[1]}/en/'
        self.strict_match = f'?strictMatch={self.api_creds.strict_match}'

        self.url_entries = self.url_ent + self.word + self.strict_match
        self.url_sentences = self.url_sent + self.word + self.strict_match
        self.lemmatizer = WordNetLemmatizer()

        self.sentences = None
        self.senses = []

    def _load_into_json(self, res: Response):
        json_output = json.dumps(res.json())
        return json.loads(json_output)

    def _lemmatize_token(self, tkn):
        if self.lemmatizer.lemmatize(tkn, pos='n') != tkn:
            return self.lemmatizer.lemmatize(tkn, pos='n')
        if self.lemmatizer.lemmatize(tkn, pos='v') != tkn:
            return self.lemmatizer.lemmatize(tkn, pos='v')
        return self.lemmatizer.lemmatize(tkn, pos='a')


    def _preprocessing(self, sentence:str, main_word:str):
        words = sentence.split()
        for idx, w in enumerate(words):
            if re.search(main_word[:3], w.lower()):
                words[idx] = main_word
        return ' '.join(words)

    def get_senses(self) -> List[Dict]:
        self.res_entries = requests.get(
            self.url_entries,
            headers={'app_id': self.api_creds.app_id, 'app_key': self.api_creds.app_key}
        )

        self.res_sentences = requests.get(
            self.url_sentences,
            headers={'app_id': self.api_creds.app_id, 'app_key': self.api_creds.app_key}
        )

        self.senses_examples = self._load_into_json(self.res_entries)
        self.sentences_examples = self._load_into_json(self.res_sentences)

        if ('results' not in self.senses_examples.keys()) or ('results' not in self.sentences_examples.keys()):
            raise ValueError(
                f'No results from the Oxford API for the word: "{self.word}"'
            )

        senses_all_res = self.senses_examples['results']
        sentences_all_res = self.sentences_examples['results']

        sense_with_examples = {}
        diff_sense_ids = []

        for res_s in sentences_all_res:
            for ent in res_s['lexicalEntries']:
                for el in ent['sentences']:
                    diff_sense_ids.append(el['senseIds'][0])

        sense_ids = set(diff_sense_ids)

        def search(id):
            for res_s in sentences_all_res:
                for ent in res_s['lexicalEntries']:
                    return [self._preprocessing(sent['text'], self.word) for sent in ent['sentences'] if sent['senseIds'][0] == id]

        for res in senses_all_res:
            for lent in res['lexicalEntries']:
                for ent in lent['entries']:
                    for idx, sens in enumerate(ent['senses']):
                        try:
                            sense_with_examples['word'] = self.word
                            sense_with_examples['sense'] = sens['id']
                            sense_with_examples['definition'] = sens['definitions'][0]

                            if 'examples' in sens.keys():
                                examples_for_senses = list(self._preprocessing(ex['text'], self.word) for ex in sens['examples'])
                            else:
                                continue

                            if sens['id'] in list(sense_ids):
                                examples_sense = search(sens['id'])
                                sense_with_examples['examples'] = list(chain(examples_sense, examples_for_senses))

                        except KeyError:
                            raise ValueError(
                                'No examples for the word: {}'.format(self.word)
                            )

                        self.senses.append(sense_with_examples.copy())
        return self.senses


if __name__ == '__main__':
    # print(OxfordDictAPI('abuse').get_senses())




    # from nltk import WordNetLemmatizer
    # lemmatizer = WordNetLemmatizer()
    sentence = 'the Welsh Marches'
    main_word = 'march'
    # def lemmatize_token(tkn):
    #     if lemmatizer.lemmatize(tkn, pos='n') != tkn:
    #         return lemmatizer.lemmatize(tkn, pos='n')
    #     if lemmatizer.lemmatize(tkn, pos='v') != tkn:
    #         return lemmatizer.lemmatize(tkn, pos='v')
    #     return lemmatizer.lemmatize(tkn, pos='a')
    #
    words = sentence.split()
    for idx, w in enumerate(words):
        if re.search(main_word[:3], w.lower()):
            words[idx] = main_word

    print(' '.join(words))