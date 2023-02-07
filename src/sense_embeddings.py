
import re
import torch
from transformers import BertTokenizer, BertModel
from transformers import logging


SENSE_EXAMPLES_FILE = '../data/target_words/senses_oxford_api.json'

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
        except ValueError:
            print(tokens)
        idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        segment_id = [1] * len(tokens)

        self.tokens_tensor = torch.tensor([idx])
        self.segments_tensors = torch.tensor([segment_id])

        with torch.no_grad():
            outputs = self.model(self.tokens_tensor, self.segments_tensors)
            hidden_states = outputs[2]

        return hidden_states[-2][0][main_token_id]


class ExtractSenseEmbeddings():
    def __init__(
            self
        ):
        self.sense = None
        self.vector_embeddings = VectorEmbeddings()

    def __call__(self, sense:dict):
        if isinstance(sense, dict):
            self.sense = sense
        else:
            raise ValueError(
                'The sense should be of instance dict'
            )
        return self

    def _infer_sentence_vector(self):
        for example in self.sense['examples']:
            yield self.vector_embeddings.infer_vector(
                doc=example,
                main_word=self.sense['word']
            )

    def infer_mean_vector(self):
        all_token_embeddings =  torch.stack(list(self._infer_sentence_vector()))
        self.sense.pop('examples', None)
        self.sense['embedding'] = torch.mean(all_token_embeddings, dim=0).tolist()

        return self.sense


def create_sense_embeddings():
    import json
    with open(SENSE_EXAMPLES_FILE) as f: all_words = json.load(f)
    sens_embedding = ExtractSenseEmbeddings()

    all_embeddings = []
    for word in all_words:
        print(f'{"-"*40} Embedding the word {word[0]["word"]} {"-"*40} ')
        output = [sens_embedding(sens).infer_mean_vector() for sens in word]
        all_embeddings.append(output)

    return all_embeddings



if __name__ == '__main__':
    import json
    with open('../embeddings/embeddings_for_senses.json', 'w') as f:
            json.dump(create_sense_embeddings(), f, indent=4)











