'''
Created on Dec 19, 2013

@author: raychen

MovieLens data set consists of:
    * 100,000 ratings (1-5) from 943 users on 1682 movies. 
    * Each user has rated at least 20 movies. 
    * Simple demographic info for the users (age, gender, occupation, zip)
'''

import os

class Recommendation():
    def __init__(self):
        self.user_rating = {}
        self.item_rating = {}
    
    def parse_data(self, fname, byUser=True):
        '''
        Parse full data set (100000 ratings), which is a tab separated list of
            user id | item id | rating | timestamp
        '''
        try:
            if not os.path.exists(fname):
                raise IOError('Error: can not find file "{}"'.format(fname))
            with open(fname, 'r') as f:
                for data in f:
                    (user_id, item_id, rating, timestamp) = data.strip().split()
                    user_id = int(user_id)
                    item_id = int(item_id)
                    rating = int(rating)
                    if byUser:      # User-based
                        if user_id not in self.user_rating.keys():
                            self.user_rating[user_id] = {}
                        self.user_rating[user_id][item_id] = rating
                    else:           # Item-based
                        if item_id not in self.item_rating.keys():
                            self.item_rating[item_id] = {}
                        self.user_rating[item_id][user_id] = rating
            if byUser:
                print('In the data set, we have {} users.'.format(len(self.user_rating)))
            else:
                print('In the data set, we have {} items.'.format(len(self.item_rating)))
        except IOError, e:
            print e
            
        def create_simple_item_based_collaborative_filtering():
            pass
            
if __name__ == '__main__':
    recommendation = Recommendation()
    recommendation.parse_data('ml-100k/u.data')