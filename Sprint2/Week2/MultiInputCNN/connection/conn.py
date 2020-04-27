from pymongo import MongoClient
import pprint

client = MongoClient("mongodb://heroku_pp4rkrx5:9919l2jg2bmm0jrvf50oj8m3f3@ds141613.mlab.com:41613/heroku_pp4rkrx5?retryWrites=false")

data_base = client['heroku_pp4rkrx5']
col = data_base['filteredTweets']

pprint.pprint(col.find_one())
