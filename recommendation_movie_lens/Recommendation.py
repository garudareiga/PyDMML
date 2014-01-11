#!/usr/bin/env python
'''
Created on Dec 19, 2013

@author: raychen

MovieLens data set consists of:
    * 100,000 ratings (1-5) from 943 users on 1682 movies. 
    * Each user has rated at least 20 movies. 
    * Simple demographic info for the users (age, gender, occupation, zip)
'''

import sys
import os
import math
import random
import csv
import time
import json
import numpy as np
from pprint import pprint
from collections import defaultdict
from pymongo import MongoClient

class Recommendation():
    def __init__(self, num_user, num_item):
        self.num_user = num_user
        self.num_item = num_item
        self.user_item_matrix = None
        self.similarity_by_item = defaultdict(list)
        self.recommendation_by_item = defaultdict(list)
    
    def get_rating(self, user_id, item_id):
        return self.user_item_matrix[user_id][item_id]
    
    def parse_data(self, fname):
        '''
        Parse full data set (100000 ratings), which is a tab separated list of
            user id | item id | rating | timestamp
        '''
        assert self.num_item > 0 and self.num_user > 0
        print('In the data set, we have {} users and {} items.'.format(self.num_user, self.num_item))
        
        # Create (943 + 1)x(1682 + 1) user-item array
        self.user_item_matrix = np.zeros((self.num_user + 1, self.num_item + 1), dtype=float)
        try:
            if not os.path.exists(fname):
                raise IOError('Error: can not find file "{}"'.format(fname))
            with open(fname, 'r') as f:
                for data in f:
                    (user_id, item_id, rating, timestamp) = data.strip().split()
                    user_id = int(user_id)
                    item_id = int(item_id)
                    rating = float(rating)
                    assert user_id <= self.num_user
                    assert item_id <= self.num_item
                    assert rating <= 5.0 and rating >= 0.0
                    self.user_item_matrix[user_id][item_id] = rating
            
            # Row 0: average rating per movie
            for item_id in xrange(1, self.user_item_matrix.shape[1]):
                ratingNum = 0
                ratingSum = 0.0
                for user_id in xrange(1, self.user_item_matrix.shape[0]):
                    if self.user_item_matrix[user_id][item_id]:
                        ratingNum += 1
                        ratingSum += self.user_item_matrix[user_id][item_id]
                self.user_item_matrix[0][item_id] = ratingSum/ratingNum
                    
            # Column 0: average rating per user
            for user_id in xrange(1, self.user_item_matrix.shape[0]):
                ratingNum = 0
                ratingSum = 0.0
                for item_id in xrange(1, self.user_item_matrix.shape[1]):
                    if self.user_item_matrix[user_id][item_id]:
                        ratingNum += 1
                        ratingSum += self.user_item_matrix[user_id][item_id]
                self.user_item_matrix[user_id][0] = ratingSum/ratingNum
        except IOError, e:
            print e
            
    def dump_csv(self, fname):
        with open(fname + '.csv', 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            csvwriter.writerow(['user\item'] + range(self.num_item))
            for n in xrange(self.user_item_matrix.shape[0]):
                csvwriter.writerow([n] + list(self.user_item_matrix[n]))              
            
    def item_similarity_pearson(self, i, j):
        '''
        Pearson's Correlation based Similarity
        
                                      covariance(i, j)
        sim(i, j) = -------------------------------------------------
                      standard_deviation(i) * standard_deviation(j)
                      
                                sum{ (R_{u,i} - avgR_{i})*(R_{u,j} - avgR_{j}) }
                  = -----------------------------------------------------------------------
                    sqrt{sum{ (R_{u,i} - avgR_{i})^2 }}*sqrt{sum{ (R_{u,j} - avgR_{j)^2 }}} 
                    
                * avgR_{i} and avgR_{j} are average ratings for each movie
                * R_{u,i} and R_{u,j} are co-rated item i and j by user u    
        '''
            
        prod_ij = 0.0
        sum_i = 0.0
        sum_j = 0.0
        
        avgRating_i = self.get_rating(0, i)
        avgRating_j = self.get_rating(0, j)
        for user_id in xrange(1, self.user_item_matrix.shape[0]):
            if self.user_item_matrix[user_id][i] > 0.0 and self.user_item_matrix[user_id][j] > 0.0:
                prod_ij += (self.get_rating(user_id, i) - avgRating_i)*\
                    (self.get_rating(user_id, j) - avgRating_j)
                sum_i += math.pow((self.get_rating(user_id, i) - avgRating_i), 2)
                sum_j += math.pow((self.get_rating(user_id, j) - avgRating_j), 2)
        
        if False:
            print('-> Calculate similarity for item {}(mean={}) & {}(mean={})'.
                  format(i, avgRating_i, j, avgRating_j))
            print('sum_i={}, sum_j={}, prod_ij={}'.format(sum_i, sum_j, prod_ij))
       
        sim = 0.0
        if prod_ij > 0.0:
            sim = prod_ij/(math.sqrt(sum_i)*math.sqrt(sum_j))
        
        return sim
        
    def item_similarity_adjusted_cosine(self, i, j):
        '''
        Adjusted Cosine Similarity
        
                      sum{ (R_{u,i} - avgR_{u})*(R_{u,j} - avgR_{u}) }
        sim(i, j) = -----------------------------------------------------------------------
                      sqrt{sum{ (R_{u,i} - avgR_{u})^2 }}*sqrt{sum{ (R_{u,j} - avgR_{u)^2 }}} 
                      
                * avgR_{u} is average rating given by user u
                * R_{u,i} and R_{u,j} are co-rated item i and j by user u    
        '''
        
        prod_ij = 0.0
        sum_i = 0.0
        sum_j = 0.0
        num_corated = 0
        for user_id in xrange(1, self.user_item_matrix.shape[0]):
            if self.user_item_matrix[user_id][i] > 0.0 and self.user_item_matrix[user_id][j] > 0.0:
                num_corated += 1
                prod_ij += (self.get_rating(user_id, i) - self.get_rating(user_id, 0))*\
                    (self.get_rating(user_id, j) - self.get_rating(user_id, 0))
                sum_i += math.pow((self.get_rating(user_id, i) - self.get_rating(user_id, 0)), 2)
                sum_j += math.pow((self.get_rating(user_id, j) - self.get_rating(user_id, 0)), 2)
                
        if num_corated == 0:
            return 0.0
        
        return prod_ij/(math.sqrt(sum_i)*math.sqrt(sum_j))
        
    def do_item_based_collaborative_filtering(self, similarity, num_similar_items=100):
        '''
        Compute similarity between two items by given a similarity measure function,
        and then sort similar items by its similarity
        '''
        
        start = time.time()
        
        print('Doing item-based collaborative filtering ...')
        for item_id_1 in xrange(1, self.user_item_matrix.shape[1]):
            if item_id_1 % 100 == 0:
                print('{:.2f}% completed'.format(float(item_id_1)/self.num_item*100))
            for item_id_2 in xrange(item_id_1 + 1, self.user_item_matrix.shape[1]):
                sim = similarity(item_id_1, item_id_2)
                if sim > 0.0:
                    self.similarity_by_item[item_id_1].append((item_id_2, sim))
                    self.similarity_by_item[item_id_2].append((item_id_1, sim))
                       
        for item_id, similar_items in self.similarity_by_item.items():
            similar_items = sorted(similar_items, key=lambda similar_item : similar_item[1], reverse=True)
            similar_items = similar_items[:num_similar_items]
            self.similarity_by_item[item_id] = similar_items
                
        #pprint(self.similarity_by_item)
        stop = time.time()                        
        print('Item-based collaborative filtering takes {:.2f} minutes'.format((stop - start)/60.0))
        
    def make_recommendations(self, user_id, num_items=10):
        '''
        Compute the prediction on an item i for a user u by computing the sum of the ratings
        given by the user on the items similar to i. Each ratings is weighted by the corresponding
        similarity s_{i,j} between items i and j.
        '''
        
        recommendation_list = []        
        for item_id in xrange(1, self.user_item_matrix.shape[1]):
            if self.user_item_matrix[user_id][item_id] > 0.0:
                continue
            
            weighted_rating_sum = 0.0
            similarity_sum = 0.0
            for similar_item_id, similarity in self.similarity_by_item[item_id]:
                weighted_rating_sum += similarity*self.user_item_matrix[user_id][similar_item_id]
                similarity_sum += similarity
            if similarity_sum > 0.0:
                prediction = weighted_rating_sum/similarity_sum
                recommendation_list.append((item_id, prediction))
            
        sorted(recommendation_list, key=lambda pred_item : pred_item[1], reverse=True)
        
        return recommendation_list[:num_items]   
            
    def save_similarity_to_json(self, fname='similarity.json'):       
        json_data = {}
        
        for item_id, similar_items in self.similarity_by_item.items():
            json_obj = {}
            
            for similar_item_id, similarity in similar_items:
                json_obj[similar_item_id] = similarity
            
            json_data[item_id] = json_obj
                
        with open(fname, 'w') as f:
            json.dump(json_data, f, indent=4)
                
    def load_similarity_from_json(self, fpath):
        if not os.path.exists(fpath):
            raise IOError('Error: can not find file "{}"'.format(fpath))
        with open(fpath, 'r') as f:
            json_data = json.load(f)
            for item_id, similar_items in json_data.items():
                self.similarity_by_item[int(item_id)] += [(int(similar_item_id), similarity) 
                                                          for similar_item_id, similarity in similar_items.items()]
            
    def save_similarity_to_mongodb(self):
        client = MongoClient("mongodb://localhost")
        db = client.pydmml
        collection = db.movielens
        try:
            for item_id, similar_items in self.similarity_by_item.items():
                document = { '_id':item_id, 'similarItems':{} }
                for similar_item_id, similarity in similar_items:
                    document['similarItems'][str(similar_item_id)] = round(similarity, 2)
                collection.insert(document)
            pass
        except:
            print "Error trying to read collection:" + sys.exc_info()[0]

    def load_similarity_from_mongodb(self):
        client = MongoClient("mongodb://localhost")
        db = client.pydmml
        collection = db.movielens
        try:
            dbIter = collection.find()
            for dbItem in dbIter:
                item_id = dbItem['_id']
                self.similarity_by_item[item_id] += [(int(similar_item_id), similarity) 
                                                          for similar_item_id, similarity in dbItem['similarItems'].items()]                    
        except:
            print "Error trying to read collection:" + sys.exc_info()[0]
    
def test1(jsonName='similarity.json'):
    ''' 
    After parsing the training set, find top-100 similar items and their corresponding similiarity score,
    and the save to a json data file.
    '''
    obj = Recommendation(num_user=943, num_item=1682)
    obj.parse_data('ml-100k/u.data')
    #obj.dump_csv('user_item_matrix')
    obj.do_item_based_collaborative_filtering(similarity=obj.item_similarity_pearson, num_similar_items=100)
    if jsonName:
        obj.save_similarity_to_json(jsonName)
    else:
        obj.save_similarity_to_mongodb() 
              
def test2(jsonName='similarity.json'):
    ''' 
    After loading top-100 similar items and their corresponding similiarity score, make a recommendation
    for a random user.
    '''
    obj = Recommendation(num_user=943, num_item=1682)
    obj.parse_data('ml-100k/u.data')
    if jsonName:
        obj.load_similarity_from_json(jsonName)
    else:
        obj.load_similarity_from_mongodb()
        
    user_id = random.randint(1, obj.num_user)
    recommendations = obj.make_recommendations(user_id)
    print('Recommendation for user ({}):'.format(user_id))
    print(recommendations)
                    
if __name__ == '__main__':
    test1()
    #test2()