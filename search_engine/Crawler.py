'''
Created on Jan 17, 2014

@author: ray
'''

import argparse
import requests
from bs4 import BeautifulSoup

class Crawler(object):
    '''
    classdocs
    '''


    def __init__(self, depth):
        '''
        Constructor
        '''
        self.urls = []
        
    def add_url(self, url):
        self.urls.append(url)
        
    def crawl(self):
        pass
    
def main():
    pass
        
if __name__ == '__main__':
    main()