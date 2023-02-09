import pandas as pd

def get_targets():
    words_file = open('../data/target_words/polysemous2.txt', 'r')
    targets = words_file.read().split('\n')

    return targets

def get_sentence(year, targets):

    results = {k: [] for k in targets}
    df = pd.read_csv('../articles_raw_data/news_' + str(year) + '.csv')
    df = df['text']

    bag = [item for sentence in df for item in sentence.split('.') if item != '']
    bag = list(filter(lambda x: (len(x.split()) > 4) and (len(x.split()) < 512), bag))



    for word in targets:
        count = 0
            
        for sentence in bag:
            if count < 1001:
                if word in sentence.split():
                    results[word].append(sentence)
                    count += 1

                else:
                    continue
            else:
                break
    
    sentences = list(set([element for v in results.values() for element in v if len(v) >= 10]))

    return sentences

def main():
    periods = [1980, 1982, 1985, 1987, 1989, 1990, 1992, 1995, 2000, 2001, 2002, 2003, 2005, 2008, 2009, 2010, 2012, 2013, 2015, 2016, 2017, 2018, 2019]
    targets = get_targets()

    for period in periods:
        print('getting sentences for: ', period)
        sentences = get_sentence(year= period, targets= targets)
        print('number of sentences: ', len(sentences))

        print('saving sentences..............................')
        file = open('../articles_raw_data/' + str(period) + '_sentences.txt','w') 
        for item in sentences:
            file.write(item+"\n")
        file.close()

if __name__ == "__main__":
    main()
