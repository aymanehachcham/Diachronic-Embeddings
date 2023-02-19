
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
from tqdm import tqdm
from transformers import BertForMaskedLM, AdamW, AutoTokenizer


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

            # if list(set(self.words) & set(doc_ids)) == []:
            #     print(list(set(self.words) & set(doc_ids)))
            #     raise AttributeError(
            #         f'None of the words: {self.words} are in the document: {idx}'
            #     )

            doc_ids[(doc_ids == self.words[0]) | (doc_ids == self.words[1]) | (doc_ids == self.words[2])] = 103
            docs[idx] = torch.tensor(doc_ids)

        return docs
    def __getitem__(self, idx):
        return dict(
            input_ids = self.encodings['input_ids'][idx].clone().detach(),
            attention_mask = self.encodings['attention_mask'][idx].detach().clone(),
            labels = self.encodings['labels'][idx].detach().clone(),
        ).copy()
    def __len__(self):
        return self.encodings.input_ids.shape[0]

class BertTrainor():
    def __init__(
            self,
            train_dataset,
            device='cpu',
            epochs=1,
        ):
        self.train_dataset = train_dataset
        self.device = device
        self.epochs = epochs

        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=16,
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

        self.model.save_pretrained("bert_model_new")

if __name__ == '__main__':

    # examples = [
    #                 "In the election of 2000, the party in effect lioa the judicial power to seize the presidency for itself, and this time the attempt succeeded.",
    #                 "He is already facing impeachment over claims that he misused public money and abuse his office since coming to power a year ago.",
    #                 "That does not make sense, that is not logical, and the judge has abuse his powers.",
    #                 "By abuse people's willingness to respond to emergencies, you make them less likely to respond to them at all.",
    #                 "Last year in Parliament, Labor's Craig Emerson accused insurance companies of abuse their market power over small smash repairers.",
    #                 "Parents are abuse the new guidelines to save money on childminding.",
    #                 "Because of their unlimited power, some consuls abuse their authority.",
    #                 "He abuse his position of power to engage in a 3-year affair with a married woman, possibly having a baby with her.",
    #                 "The list itself was prefaced with the following insight: \u2018Leaders with absolute power too often abuse it.\u2019",
    #                 "Today, we understand that the era of political ignorance is over and that those in power who abuse their authority can be challenged and held liable in a court of law."
    #             ]


    with open('../data/all_sentences.txt', 'r') as f:
        sentences = f.read()

    samples = sentences.split('\n')[:100]
    data = NewsExamplesDataset(samples, chunk_size=420, words_to_mask=["abuse", "fight", "market"])
    BertTrainor(data, device='cpu', epochs=10).train()

