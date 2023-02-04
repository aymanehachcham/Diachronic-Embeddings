
import torch
from transformers import BertTokenizer, BertModel
from transformers import logging


SENSE_EXAMPLES_FILE = '../data/target_words/senses_oxford_api.json'

sense_ex = {
            "word": "state",
            "sense": "m_en_gbus0989430.006",
            "definition": "the particular condition that someone or something is in at a specific time",
            "examples": [
                "And final confirmation of my poor state of mind from lack of sleep came when Mark returned from going out.",
                "At times she is combative, at times submissive, according to the situation and her state of mind.",
                "A positive state of mind is also thought to be of great help in protecting against such problems.",
                "The cowboy is the archetypal American hero, and the western fits America's current state of mind.",
                "Hopefully, by moving to the north for a little while, my work will improve and so will my state of mind.",
                "His state of mind becomes even more troubled when a copy of Rebecca's childhood diary arrives anonymously in the post.",
                "He will under go a psychiatric examination to determine his state of mind at the time of the killings, he said.",
                "After that initial catharsis had passed she asked me to fill in some questionnaires so that she could establish my state of mind.",
                "Unfortunately, in her state of mind she'd forgotten that she had worn a black jacket that night.",
                "Her research suggests that a little after-work light could lead to being in a better state of mind.",
                "There is a parallel between his state of mind in the late 1960s and when he wrote the book in the early 1940s.",
                "I think you've managed to capture my state of mind pretty much exactly.",
                "Last week's ITV documentary raised serious questions about his state of mind.",
                "They have also begun examining his computer for clues as to his state of mind and any friends who might not have been known to his parents.",
                "Lately, I haven't really been in the right state of mind to make decisions.",
                "I needed to hear words that only he could say, words that would shake me out of my unsettled state of mind.",
                "Astaphan said a critical point was the state of mind of the defendant when he made the statements.",
                "I love colour and use it to represent my state of mind - green is my favourite.",
                "Hopefully I'll have surfaced by Sunday afternoon and will be in a fit state to drive over and pick up the family.",
                "I have already seen it three times and each time I gain new insights into my own state of mind.",
                "the state of the company's finances",
                "we're worried about her state of mind"
            ]
        }


class VectorEmbeddings():
    def __init__(
        self
    ):
        self._tokens = []
        self.model = None
        self.vocab = False
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

    def infer_vector(self, doc:str):
        if self.vocab:
            marked_text = "[CLS] " + doc + " [SEP]"
            tokens = self.bert_tokenizer.tokenize(marked_text)
            idx = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            segment_id = [1] * len(tokens)

            self.tokens_tensor = torch.tensor([idx])
            self.segments_tensors = torch.tensor([segment_id])

            with torch.no_grad():
                outputs = self.model(self.tokens_tensor, self.segments_tensors)
                hidden_states = outputs[2]

            return torch.mean(hidden_states[-2][0], dim=0)

        else:
            raise ValueError(
                'The Embedding model has not been initialized'
            )

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
            yield self.vector_embeddings.infer_vector(example)

    def infer_mean_vector(self):
        all_embeddings =  torch.stack(list(self._infer_sentence_vector()))
        self.sense.pop('examples', None)
        self.sense['embedding'] = torch.mean(all_embeddings, dim=0).tolist()

        return self.sense


def create_sense_embeddings():
    import json
    with open(SENSE_EXAMPLES_FILE) as f: all_words = json.load(f)
    sens_embedding = ExtractSenseEmbeddings()

    all_embeddings = []
    for word in all_words:
        output = [sens_embedding(sens).infer_mean_vector() for sens in word]
        all_embeddings.append(output)

    return all_embeddings



if __name__ == '__main__':
    # print(ExtractSenseEmbeddings().__call__(sense_ex).infer_mean_vector())
    import json
    with open('../data/target_words/senses_embeddings.json', 'w') as f:
            json.dump(create_sense_embeddings(), f, indent=4)











