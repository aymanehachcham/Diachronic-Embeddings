

import torch
import json
from transformers import BertTokenizer, BertModel
from transformers import logging
from settings import EmbeddingFiles
from components import SenseEmbedding, OxfordAPIResponse


SENSE_EXAMPLES_FILE = '../data/target_words/senses_oxford_api.json'

example = {
                "id": "m_en_gbus0822990.005",
                "definition": "intended to protect someone or something",
                "examples": [
                    "Other male contraceptive formulations in protective have been shown to decrease good HDL cholesterol levels, which are protective against heart disease.",
                    "And that's not to mention research showing that beer is protective against gallstone formation, osteoporosis and even diabetes.",
                    "Although carbohydrates boosted blood sugar, protective protective stable glucose levels that may be more protective against afternoon tiredness.",
                    "But there is no treatment known to eliminate them, although the antioxidants in fruit and vegetables are protective against many things.",
                    "So in terms of making the country work to be protective against these kinds of things, we have a way to go.",
                    "However, we do also know that activity is good for you, and even protective against some of the more terminal conditions that we can get, like the aforementioned heart attacks.",
                    "In epidemiologic studies, whole grains, vegetables, and fruits are often more protective against diseases than fibre supplements.",
                    "Antioxidant micronutrients found in fruits and vegetables have been shown in numerous studies to be protective against cancer.",
                    "Surely, it was an inconvenient oddity - the thin silk kimonos favored by geisha were more decorative than protective against the elements.",
                    "They are calling for protective barriers to be put in place and intend to raise awareness on this issue at the ceremony.",
                    "He added: \u2018Anyone moving sandbags should do so carefully as they are heavy, and people should wear protective gloves.\u2019",
                    "One day I opened up a disposable camera package in my classroom, which came in an anti-x-ray metal foil packet and was wrapped in a cardboard protective lining.",
                    "Evidently, whatever sort of debilitating bug was on that disk, it had so far managed to get past one of the best protective protective available.",
                    "Elevation and windage are adjustable by removing the two protective covers over the setscrews on the front of the sight housing.",
                    "The blade body consists of a spar assembly, leading edge protective strips, skins over a honeycomb core and a trailing edge strip.",
                    "Affluent populations are, in general, the first to take up practices that are perceived as protective of child health; in the latter part of the decade, this meant declining immunisation.",
                    "A diet rich in antioxidant vitamins (A, C and E), available in fresh fruit and vegetables, is thought to be protective against stomach cancer.",
                    "These protective zones also surround public and private vehicles.",
                    "As he arrived he saw neighbours being forced back from the house by the intense heat of the inferno, and dashed towards the end terrace house without stopping to put on any protective protective clothing or his breathing gear.",
                    "protective gloves are worn to minimize injury"
                ]
            }

class VectorEmbeddings():
    def __init__(
        self
    ):
        self._tokens = []
        self.model = None
        self.vocab = False
        self.lematizer = None

        logging.set_verbosity_error()
        self._bert_case_preparation()

    @property
    def tokens(self):
        return self._tokens

    def _bert_case_preparation(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained(
            'bert-base-uncased',
            output_hidden_states = True,
        )
        self.model.eval()
        self.vocab = True

    def infer_vector(self, doc:str, main_word:str):
        if not self.vocab:
            raise ValueError(
                'The Embedding model has not been initialized'
            )
        marked_text = "[CLS] " + doc + " [SEP]"
        tokens = self.bert_tokenizer.tokenize(marked_text)
        try:
            main_token_id = tokens.index(main_word)
            idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self.tokens_tensor = torch.tensor([idx])
            self.segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self.model(self.tokens_tensor, self.segments_tensors)
                hidden_states = outputs[2]

            return hidden_states[-2][0][main_token_id]

        except IndexError:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the list of tokens: {tokens}'
            )


class ExtractSenseEmbeddings():
    def __init__(
            self
        ):
        self.sense = None
        self.word = None
        self.vector_embeddings = VectorEmbeddings()
        self.api_component = OxfordAPIResponse()

    def __call__(self, sense:dict, main_w):
        if not isinstance(sense, dict):
            raise ValueError(
                f'Expected type dict for the sense, but got type: {type(sense)}'
            )
        self.sense = sense
        self.word = main_w
        return self

    def _infer_sentence_vector(self):
        for example in self.sense['examples']:
            yield self.vector_embeddings.infer_vector(
                doc=example,
                main_word=self.word
            )

    def infer_mean_vector(self):
        all_token_embeddings =  torch.stack(list(self._infer_sentence_vector()))
        self.sense.pop('examples', None)
        self.sense['embedding'] = torch.mean(all_token_embeddings, dim=0).tolist()

        return self.sense


def create_sense_embeddings():
    files = EmbeddingFiles()
    with open(files.oxford_word_senses) as f: all_words = json.load(f)
    sens_embedding = ExtractSenseEmbeddings()

    all_embeddings = []
    for word in all_words:
        print(f'{"-"*40} Embedding the word {word["word"]} {"-"*40} ')
        word['senses'] = [sens_embedding(sens, word["word"]).infer_mean_vector() for sens in word['senses']]
        all_embeddings += [word.copy()]

    return all_embeddings



if __name__ == '__main__':
    import json
    with open('../embeddings/embeddings_for_senses.json', 'w') as f:
            json.dump(create_sense_embeddings(), f, indent=4)











