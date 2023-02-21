
import torch
from transformers import BertTokenizer, BertModel
from transformers import logging
from components import OxfordAPIResponse
from settings import FileLoader
import json


class VectorEmbeddings():
    """
    This class is used to infer the vector embeddings of a word from a sentence.

    Methods
    -------
        infer_vector(doc:str, main_word:str)
            This method is used to infer the vector embeddings of a word from a sentence.
        _bert_case_preparation()
            This method is used to prepare the BERT model for the inference.
    """
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

    # Generate the vector embeddings for the main word in the sentence
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

        except ValueError:
            raise ValueError(
                f'The word: "{main_word}" does not exist in the list of tokens: {tokens} from {doc}'
            )


class ExtractSenseEmbeddings():
    """
    Wrapper class for the Vector embeddings that is used to extract the embeddings for all the senses.

    Attributes
    ----------
        sense: dict
            The sense for which the embeddings are to be extracted.
        word: str
            The word for which the embeddings are to be extracted.
        vector_embeddings: VectorEmbeddings
            The class that is used to extract the embeddings.
        api_component: OxfordAPIResponse
            The class that is used to extract the senses.
        all_words: list
            The list of all the words for which the embeddings are to be extracted.

    Methods
    -------
        __call__(sense:dict, main_w:str)
            This method is used to initialize the object.
        _infer_sentence_vector()
            This method is used to infer the vector embeddings for all the sentences in the sense.
        infer_mean_vector()
            This method is used to infer the mean vector embeddings for all the sentences in the sense.
        create_sense_embeddings()
            This method is used to extract the embeddings for all the senses.

    """
    def __init__(
            self
        ):
        self.sense = None
        self.word = None
        self.vector_embeddings = VectorEmbeddings()
        self.api_component = OxfordAPIResponse()
        self.all_words = FileLoader.load_files(self.__class__.__name__)

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

    def create_sense_embeddings(self):
        all_embeddings = []
        for word in self.all_words:
            print(f'{"-"*40} Embedding the word {word["word"]} {"-"*40} ')
            word['senses'] = [self(sens, word["word"]).infer_mean_vector() for sens in word['senses']]
            all_embeddings += [word.copy()]

        return all_embeddings



if __name__ == '__main__':
    e = ExtractSenseEmbeddings()
    with open('../embeddings/embeddings_for_senses.json', 'w') as f:
            json.dump(e.create_sense_embeddings(), f, indent=4)











