

if __name__ == '__main__':
    from api_calls.utils import perform_on_all_words
    import json

    POLYSEMOUS = '../data/target_words/polysemous.txt'

    ## Generate the senses and examples for each word of the list polysemous
    with open('../../data/target_words/senses_oxford_api.json', 'w') as f:
        json.dump(perform_on_all_words(POLYSEMOUS), f, indent=4)
