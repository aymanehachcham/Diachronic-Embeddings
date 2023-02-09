from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


def get_targets():
    words_file = open('../data/target_words/polysemous.txt', 'r')
    targets = words_file.read().split('\n')

    return targets

def lemmatize(words):
    lmtzr = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    newstopwords = ['mrs','mr','say','says','said','tell','told','seem','seems','seemed','ask','asks','asked','upon','aboard','about','above','account','across','after','against','ago','ahead','along','alongside','amid','among','around','aside','at','atop','away','because','before','behalf','behind','below','beneath','beside','besides','between','beyond','but','by','circa','considering','depending','despite','down','due','during','for','from','further','given','in','including','inside','instead','into','less','like','near','notwithstanding','of','off','on','onto','opposite','other','out','outside','over','owing','per','plus','regarding','regardless','round','since','than','thanks','through','throughout','till','to','together','toward','towards','under','underneath','unlike','until','up','upon','versus','via','with','within','without']

    targets = list(filter(lambda x: not x in newstopwords, words))
    targets = list(filter(lambda x: not x in stop_words, targets))
    targets = list(map(lambda x: lmtzr.lemmatize(x, pos ="v"), words))
    targets = list(filter(lambda x: len(x) > 5, targets))

    return targets

def main():
    targets = get_targets()
    targets = lemmatize(targets)
    targets = list(set(targets))
    
    print('number of target words: ', len(targets))

    file = open('../data/target_words/polysemous2.txt','w') 
    for item in targets:
        file.write(item+"\n")
    file.close()

    print('target words saved to: ../data/target_words/polysemous2.txt')

    return targets

if __name__ == "__main__":
    main()