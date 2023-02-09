from transformers import BertTokenizer, BertModel
import torch

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
model.eval()

def get_targets():
    words_file = open('../data/target_words/polysemous2.txt', 'r')
    targets = words_file.read().split('\n')

    return targets

def get_sentences(year, targets):
    with open() as f:
        sentences= []
    return sentences


def infer_vector(doc:str):

    marked_text = "[CLS] " + doc + " [SEP]"
    tokens = bert_tokenizer.tokenize(marked_text)[:512]
    idx = bert_tokenizer.convert_tokens_to_ids(tokens)
    segment_id = [1] * len(tokens)


    tokens_tensor = torch.tensor([idx])
    segments_tensors = torch.tensor([segment_id])

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    hidden_states = hidden_states[-2][0]

    return hidden_states, tokens


def get_embed(sentences, targets):


    results = {k: {'word': k, 'sentence_number_index': [] , 'embeddings': []} for k in targets}

    for i in range(len(sentences)): #len(sentences)
        if i%1000 == 0:
            print(i)

        embeddings, tokens = infer_vector(sentences[i])

        for word in targets:
            if len(results[word]['sentence_number_index']) < 1000:
                if word in tokens:
                    index = tokens.index(word)
                    embedding = embeddings[index].tolist()

                    results[word]['sentence_number_index'].append([i, index])
                    results[word]['embeddings'].append(embedding)
                    
                else:
                    continue
            else:
                continue

    return results

def get_all(year):
    print('getting targets ..............................')
    targets = get_targets()
    print('getting sentences ............................')
    sentences = get_sentences(year= year, targets= targets)
    print('number of sentences: ', len(sentences))

    print('saving sentences..............................')
    file = open('../articles_raw_data/' + str(year) + '_sentences.txt','w') 
    for item in sentences:
        file.write(item+"\n")
    file.close()
    
    print('getting embeddings for sentences ..............')
    results = get_embed(
        sentences= sentences,
        targets= targets
    )
    print('got embeddings  ..............................')
    file = []

    for word in targets:
        file.append(results[word])
    
    print('saving embeddings ............................')
    import json
    with open('../embeddings/embeddings_' + str(year) + '.json', "w") as final:
        json.dump(file, final, indent= 4)
    
    print(year, 'done ............................')
    
def main():
    periods = [1980, 1982, 1985, 1987, 1989, 1990, 1992, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012, 2013, 2015, 2016, 2017, 2018, 2019]
    for period in periods:
        print(period)
        get_all(period)


if __name__ == "__main__":
    main()