import random


def main():
    words_file = open('../data/target_words/polysemous2.txt', 'r')
    targets = words_file.read().split('\n')

    new_targets = random.choices(targets, k=100)

    file = open('../data/target_words/polysemous3.txt','w') 
    for item in new_targets:
        file.write(item+"\n")
    file.close()

    print('finished')


if __name__ == '__main__':
    main()