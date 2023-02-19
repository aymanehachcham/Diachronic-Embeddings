
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
from transformers import BertForMaskedLM, AdamW


class NewsExamplesDataset(Dataset):
    def __init__(self,
                 examples:List[str],
                 words_to_mask: List[str],
                 chunk_size=419,

        ):
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = self.tokenizer(self.examples, return_tensors='pt', max_length=chunk_size, truncation=True, padding="max_length")
        self.encodings["word_ids"] = [self.encodings.word_ids(i) for i in range(len(self.encodings["input_ids"]))]
        self.encodings["labels"] = self.encodings["input_ids"].detach().clone()
        self.words = self.tokenizer.convert_tokens_to_ids(words_to_mask)

        self.masked_embeddings = self._pick_masked_embeddings(self.encodings['input_ids'])
        self.max_length = chunk_size

    def _pick_masked_embeddings(self, inputs:torch.Tensor, percentage_size=0.5):
        dim = inputs.shape[0]
        idx = np.random.choice(list(range(dim)), int(dim * percentage_size), replace=False)
        docs = inputs[idx]

        for idx, doc in enumerate(docs):
            doc_ids = doc.numpy().flatten()
            doc_ids[(doc_ids == self.words[0]) | (doc_ids == self.words[1]) | (doc_ids == self.words[2])] = 103
            docs[idx] = torch.tensor(doc_ids)

        return docs
    def __getitem__(self, idx):
        return dict(
            input_ids = self.encodings['input_ids'][idx].clone().detach(),
            attention_mask = self.encodings['attention_mask'][idx].detach().clone(),
            # word_ids = torch.tensor(self.encodings['word_ids'][idx]),
            labels = self.encodings['labels'][idx].detach().clone(),
        )
    def __len__(self):
        return self.encodings.input_ids.shape[0]

class BertTrainor():
    def __init__(
            self,
            train_dataset,
            device='cpu',
            epochs=1,
        ):
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.train_dataset = train_dataset
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.device = device
        self.epochs = epochs

        self.model.train()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=3,
            shuffle=True,
        )

    def train(self):
        for epoch in range(self.epochs):
            loop = tqdm(self.train_dataloader, leave=True)
            for batch in loop:
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # process
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                # print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())

        self.model.save_pretrained("bert_model")

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import torch

    examples = [
                    "In the election of 2000, the party in effect abuse the judicial power to seize the presidency for itself, and this time the attempt succeeded.",
                    "He is already facing impeachment over claims that he misused public money and abuse his office since coming to power a year ago.",
                    "That does not make sense, that is not logical, and the judge has abuse his powers.",
                    "By abuse people's willingness to respond to emergencies, you make them less likely to respond to them at all.",
                    "Last year in Parliament, Labor's Craig Emerson accused insurance companies of abuse their market power over small smash repairers.",
                    "Parents are abuse the new guidelines to save money on childminding.",
                    "Because of their unlimited power, some consuls abuse their authority.",
                    "He abuse his position of power to engage in a 3-year affair with a married woman, possibly having a baby with her.",
                    "The list itself was prefaced with the following insight: \u2018Leaders with absolute power too often abuse it.\u2019",
                    "Today, we understand that the era of political ignorance is over and that those in power who abuse their authority can be challenged and held liable in a court of law."
                ]


    data = NewsExamplesDataset(examples, chunk_size=420, words_to_mask=["abuse", "fight", "market"])
    BertTrainor(data, device='cpu', epochs=1).train()

    model = BertForMaskedLM.from_pretrained("bert_model")
    print(model)

    # print(pick_15_percent(d.encodings['input_ids']))


    # def group_by_chunks(samples, chunk_size=420):
    #     all_samples = []
    #     for sample in samples:
    #         tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    #         result = tokenizer(sample, return_tensors='pt', max_length=chunk_size, truncation=True, padding="max_length")
    #         result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    #         result.pop("token_type_ids")
    #         result["labels"] = result["input_ids"].clone()
    #
    #         all_samples += [result.copy()]
    #
    #     return all_samples
    #
    #
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # words = tokenizer.convert_tokens_to_ids(["abuse", "fight", "market"])
    # res = group_by_chunks(examples)
    # inputs = res[0]["input_ids"].numpy().flatten()
    #
    # rand = torch.rand(inputs.shape)
    #
    # inputs[(inputs == words[0]) | (inputs == words[1]) | (inputs == words[2])] = 103
    # print(inputs)

