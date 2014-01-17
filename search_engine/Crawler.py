'''
Created on Jan 17, 2014

@author: ray
'''

import argparse
import requests
import Queue
import logging
from bs4 import BeautifulSoup

web_crawl_logger = logging.getLogger(__name__)
handler_stream = logging.StreamHandler()
web_crawl_logger.addHandler(handler_stream)
web_crawl_logger.setLevel(logging.DEBUG)

class WebPage:
    def __init__(self, url, depth):
        self.url = url
        self.depth = depth

class Crawler:
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.depth = 0
        
    def crawl(self, urls, max_depth):
        web_page_queue = Queue.Queue()
        for url in urls:
            web_page = WebPage(url, 0)
            web_page_queue.put(web_page)
        
        while not web_page_queue.empty():
            # parse_web_page
            web_page = web_page_queue.get()
            try:
                r = requests.get(web_page.url)
                #if r.status_code != requests.codes.OK:
                #    raise
                web_crawl_logger.info('Get {}'.format(web_page.url))
                soup = BeautifulSoup(r.content)
                self.parseText(soup)
                
                # Stop crawling on the current web page
                if web_page.depth == max_depth:
                    continue
                # Continue crawling on the current web page
                for a in soup.find_all('a'):
                    next_web_page = WebPage(a.attrs['href'], web_page.depth + 1) 
                    web_page_queue.put(next_web_page)      
            except (KeyboardInterrupt, SystemExit):
                raise
            #except Exception as e:
            except:
                web_crawl_logger.warn('Can *not* open url {}'.format(web_page.url))
                continue
            
    
    def parseText(self, soup):
        pass
    
def main():
    parser = argparse.ArgumentParser(description='A Simple Web Crawler')
    parser.add_argument('--url', dest='urls', action='append',
                       default=[], required=True,
                       help='Add url to start crawl with',)
    parser.add_argument('--depth', dest='depth', action='store',
                       default=0, required=False,
                       help='The link depth to crawl')
    args = parser.parse_args()
    
    urls = args.urls
    depth = args.depth
    crawler = Crawler()
    crawler.crawl(urls, depth)
        
if __name__ == '__main__':
    main()
