# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:11:02 2018

@author: Mohammadsuzan.Shaikh
"""

import re
import tweepy
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
import nltk

read='E:\\My Python codes\\Twitter sentiment analysis\\'

consumer_key = 'ttjoXYSoVqbhogrgoMu77QuNA'
consumer_secret = 'HnsCycPBrREdg0cZi9IsoPravhYyPZIDvOeDumn5OWjNv6LnPG'
access_token = '3060807680-cR996jc0aUX5qHjRVG3t4r4bjq4FdyoAGO3QcDT'
access_token_secret = '68Kv14Kapn1Ad9Q4EtCyrzEpiTvOdIStlEBdR4PzunKXj'
        
#class StdOutListener(tweepy.StreamListener):
#        
#    def on_data(self, data):
#        print data
#        return True
#        
#    def on_error(self, status):
#        print status
#
#
#if __name__ == '__main__':
#
#    #This handles Twitter authetification and the connection to Twitter Streaming API
#    l = StdOutListener()
#    auth = OAuthHandler(consumer_key, consumer_secret)
#    auth.set_access_token(access_token, access_token_secret)
#    stream = tweepy.Stream(auth, l)
#
#    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
#    tweets_data=stream.filter(track=['justiceforasifa'])
    

'''Tweets for particular user'''
#def get_all_tweets(screen_name):
#    #Twitter only allows access to a users most recent 3240 tweets with this method
#
#    #authorize twitter, initialize tweepy
#    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#    auth.set_access_token(access_token, access_token_secret)
#    api = tweepy.API(auth)
#
#    #initialize a list to hold all the tweepy Tweets
#    alltweets = []  
#
#    #make initial request for most recent tweets (200 is the maximum allowed count)
#    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
#
#    #save most recent tweets
#    alltweets.extend(new_tweets)
#
#    #save the id of the oldest tweet less one
#    oldest = alltweets[-1].id - 1
#
#    #keep grabbing tweets until there are no tweets left to grab
#    while len(new_tweets) > 0:
#        print "getting tweets before %s" % (oldest)
#
#        #all subsiquent requests use the max_id param to prevent duplicates
#        new_tweets = api.user_timeline(screen_name = 'sachin_rt',count=200,max_id=oldest)
#
#        #save most recent tweets
#        alltweets.extend(new_tweets)
#
#        #update the id of the oldest tweet less one
#        oldest = alltweets[-1].id - 1
#
#        print "...%s tweets downloaded so far" % (len(alltweets))
#
#    #transform the tweepy tweets into a 2D array that will populate the csv 
#    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
#
#    #write the csv  
#    return outtweets
#
#
#if __name__ == '__main__':
#    #pass in the username of the account you want to download
#    get_all_tweets("HarvardBiz")

''''''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api=tweepy.API(auth)

tweets=[]

for tweet in tweepy.Cursor(api.search,q='#JusticeForAsifa',count=1000,lang='en',since='2018-01-01',wait_on_rate_limit=True).items():
                           print(tweet)
                           #twt=tweet
                           #time.sleep(60)
                           if hasattr(tweet,'retweeted_status')==False:
                               tweets.append({'id':tweet.id,
                                              'screen_name':tweet.user.screen_name,
                                              'user_name':tweet.user.name,
                                              'verified_acct':tweet.user.verified,
                                              'favorite_count':tweet.favorite_count,
                                              'truncated_tweet':tweet.truncated,
                                              'tweet':tweet.text,
                                              'created_at':tweet.created_at,
                                              'status_count':tweet.user.statuses_count,
                                              'followers_count':tweet.user.followers_count,
                                              'freinds_count':tweet.user.friends_count,
                                              'retweet_count':tweet.retweet_count,
                                              'hash_tag':[d.get('text') for d in tweet.entities['hashtags']],
                                              'user_mentions':[d.get('text') for d in tweet.entities['user_mentions']]}.copy())
                           else:
                               tweets.append({'id':tweet.id,
                                              'screen_name':tweet.user.screen_name,
                                              'user_name':tweet.user.name,
                                              'verified_acct':tweet.user.verified,
                                              'favorite_count':tweet.favorite_count,
                                              'truncated_tweet':tweet.truncated,
                                              'tweet':tweet.text,
                                              'created_at':tweet.created_at,
                                              'status_count':tweet.user.statuses_count,
                                              'followers_count':tweet.user.followers_count,
                                              'freinds_count':tweet.user.friends_count,
                                              'retweet_count':tweet.retweet_count,
                                              'hash_tag':[d.get('text') for d in tweet.entities['hashtags']],
                                              'user_mentions':[d.get('text') for d in tweet.entities['user_mentions']],
                                              'original_tweeter':tweet.retweeted_status.user.name}.copy())

tweets['original_tweeter'].value_counts()

tweets=pd.DataFrame(tweets)

tweets=pd.read_excel(read+'\Data\Asifa total tweets.xlsx')

'''Check for any url in tweet'''
def find_url(string):
    # findall() has been used 
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return len(url)

def remove_urls(string):
    string = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', string, flags=re.MULTILINE)
    return(string)

def remove_non_alphanumeric(string):
    string = re.sub(r'([^\s\w]|_)+', '', string, flags=re.MULTILINE)
    return(string)

def text_after_colons(string):
    return string[string.find(':')+1:len(string)]
    
tweets['presence_of_url']=tweets['tweet'].apply(lambda x: find_url(x))
tweets['presence_of_url']=np.where(tweets['presence_of_url']>0,1,0)
tweets.to_excel(read+'Data\\Asifa total tweets.xlsx')
tweets['tweet']=tweets['tweet'].apply(lambda x:remove_urls(x))
tweets['tweet']=tweets['tweet'].apply(lambda x:remove_non_alphanumeric(x).strip())
tweets['tweet']=tweets['tweet'].apply(lambda x:text_after_colons(x))

non_duplicated=tweets[~tweets['tweet'].duplicated()]

sum(tweets['tweet'].apply(lambda x:x.lower()).str.contains('narendra modi|namo|narendramodi|modi'))/float(len(tweets))

'''Text preprocessing'''
tweets_texts = non_duplicated["tweet"].tolist()
stopwords=stopwords.words('english')
english_vocab = set(w.lower() for w in nltk.corpus.words.words())

def process_tweet_text(tweet):
   if tweet.startswith('@null'):
       return "[Tweet not available]"
   tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
   tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
   tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
   twtok = TweetTokenizer(strip_handles=True, reduce_len=True)
   tokens = twtok.tokenize(tweet)
   tokens = [i.lower() for i in tokens if i not in stopwords and len(i) > 2 and  
                                             i in english_vocab]
   return tokens
 
words = []
for tw in tweets_texts:
     words += process_tweet_text(tw)

word_freq=pd.Series(words).value_counts()

'''N-Gram approach'''
trigram_measures = nltk.collocations.TrigramAssocMeasures()
finder = TrigramCollocationFinder.from_words(words, 5)
finder.apply_freq_filter(5)
print(finder.nbest(trigram_measures.likelihood_ratio, 20))

'''Clustering'''
cleaned_tweets = []
for tw in tweets_texts:
    words = process_tweet_text(tw)
    cleaned_tweet = " ".join(w for w in words if len(w) > 2 and w.isalpha()) #Form sentences of processed words
    cleaned_tweets.append(cleaned_tweet)
non_duplicated['CleanTweetText'] = cleaned_tweets

from sklearn.feature_extraction.text import TfidfVectorizer  
tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))  
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_tweets)  
feature_names = tfidf_vectorizer.get_feature_names() # num phrases  
from sklearn.metrics.pairwise import cosine_similarity  
dist = 1 - cosine_similarity(tfidf_matrix)  
print(dist) 

from sklearn.cluster import KMeans  
num_clusters = 3  
km = KMeans(n_clusters=num_clusters)  
km.fit(tfidf_matrix)  
clusters = km.labels_.tolist()  
non_duplicated['ClusterID'] = clusters  
print(non_duplicated['ClusterID'].value_counts())

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster {} : Words :".format(i))
    for ind in order_centroids[i, :20]: 
        print(' %s' % feature_names[ind])