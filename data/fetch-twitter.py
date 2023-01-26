import tweepy 
from timeit import default_timer as timer

class TWEETS():

    def __init__(self, 
                 api_key: str='ZehWP8rFPjoxQ12cVRW9vFIyC',
                 api_secret_key: str='qeYqfRNcKAsCCFCqBgFSp8nNIHs6fkqfPTD0o9CCDQQWoTNcyy',
                 bearer_token: str='AAAAAAAAAAAAAAAAAAAAAHHIlQEAAAAAq9%2BeFsnNJ4eLojGrlKjjwjIQMa0%3Dj0nAHdiPhMiQxcuxsGUCiCkMpY8vDiRqqigtF2sgIIC07mxAbC',
                 access_token = '1156290383430766593-tooYLBPcBGHxxGrHm8358Q2Ob93bMB',
                 access_token_secret = '2a7b4YI3rJgNG9iB9XklhIvKWaPafueBvbVGNOtCZ597p'
                ):
        
        self.api_key = api_key
        self.api_secret_key = api_secret_key
        self.bearer_token = bearer_token
        self.access_token = access_token
        self.access_token_secret = access_token_secret
    
        self.auth = tweepy.OAuth1UserHandler(self.api_key, self.api_secret_key, self.access_token, self.access_token_secret)

        self.api = tweepy.API(self.auth, wait_on_rate_limit = True , wait_on_rate_limit_notify = True)


    def fetch(self, api = ''):

        count = 0
        start = timer()

        # Fetch tweets from the twitter API using the following loop:
        list_of_tweets = []
        
