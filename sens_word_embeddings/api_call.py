import requests
from requests.models import Response
import json
from typing import List, Dict
from itertools import chain


class OxfordDictAPI():
    def __init__(
            self,
            word_id: str
    ):

        self.headers = {
            "Accept": "application/json",
            "app_id": "8beabadc",
            "app_key": "4d85f3e2e7cd293da9a811f156c99841"
        }

        if isinstance(word_id, str):
            self.word = word_id
        else:
            raise ValueError(
                'The word id should be a string'
            )

        self.query = ('entries', 'sentences')
        self.url_ent = 'https://od-api.oxforddictionaries.com/api/v2/{}/en/'.format(self.query[0])
        self.url_sent = 'https://od-api.oxforddictionaries.com/api/v2/{}/en/'.format(self.query[1])
        self.strict_match = '?strictMatch=true'

        self.url_entries = self.url_ent + self.word + self.strict_match
        self.url_sentences = self.url_sent + self.word + self.strict_match

        self.sentences = None
        self.senses = []

    def _load_into_json(self, res: Response):
        json_output = json.dumps(res.json())
        return json.loads(json_output)

    def search(self, id):
        if self.sentences is not None:
            return list(sent['text'] for sent in self.sentences if sent['senseIds'][0] == id)

    def get_senses(self) -> List[Dict]:
        self.res_entries = requests.get(
            self.url_entries,
            headers={'app_id': self.headers['app_id'], 'app_key': self.headers['app_key']}
        )

        self.res_sentences = requests.get(
            self.url_sentences,
            headers={'app_id': self.headers['app_id'], 'app_key': self.headers['app_key']}
        )

        self.senses_examples = self._load_into_json(self.res_entries)
        self.sentences_examples = self._load_into_json(self.res_sentences)
        api_call_senses = self.senses_examples['results'][0]['lexicalEntries'][0]['entries'][0]['senses']
        sentences = self.sentences_examples['results'][0]['lexicalEntries'][0]['sentences']

        try:
            self.senses_examples['results']
        except KeyError:
            raise ValueError(
                'No resutls for senses'
            )
        try:
            self.sentences_examples['results']
        except KeyError:
            raise ValueError(
                'No resutls for senteces'
            )

        sense_with_examples = {}
        diff_sense_ids = []

        for el in sentences:
            diff_sense_ids.append(el['senseIds'][0])

        sense_ids = set(diff_sense_ids)

        def search(id):
            return list(sent['text'] for sent in sentences if sent['senseIds'][0] == id)

        for idx, sens in enumerate(api_call_senses):
            try:
                sense_with_examples['word'] = self.word
                sense_with_examples['sense'] = sens['id']
                sense_with_examples['definition'] = sens['definitions'][0]
                examples_for_senses = list(ex['text'] for ex in sens['examples'])

                if sens['id'] in list(sense_ids):
                    examples_sense = search(sens['id'])
                    sense_with_examples['examples'] = list(chain(examples_sense, examples_for_senses))
            except KeyError:
                raise ValueError(
                    'No examples for the word: {}'.format(self.word)
                )

            self.senses.append(sense_with_examples.copy())
        return self.senses