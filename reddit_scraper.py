import requests
from bs4 import BeautifulSoup
import time
import random
import csv
from initial_data_loading import load_discourse_data
from secrets/reddit_related import login_details


class Reddit_Scraper(object):
    """
    To solve a problem I didn't know I had
    TODO: datetime handling, what goes in init?
    """
    def __init__(self, date_file='list_of_dates', error_file='error_log'):
        self.date_file = date_file
        self.error_file = error_file

    def login(self, login_details, user_agent):
        #initialize praw instance for retrieving data
        self.reddit = praw.Reddit(client_id=login_details['CLIENT_ID'],
                     client_secret=login_details['CLIENT_SECRET'],
                     password=login_details['PASSWORD'],
                     user_agent=user_agent,
                     username=login_details['USERNAME'])

    def _gen_submission(self, url):
        with open(self.error_file, 'a') as f:
            try:
                req = self.reddit.submission(url=url)
            except:
                writer = csv.writer(f)
                writer.writerow("failed to return {url}".format(url))
        return req

        if
    def get_dates(self, url_list, time_delay=1.5, rand_time=False):
        """
        INPUTS:
        url_list (list of strings) - urls to scrape
        time_delay (int) - how long to wait in between urls (in seconds). Defaults to 1.5 seconds. Reddit API is max 60 reqs/min.
        rand_time - whether or not to randomize the time in between requests
        OUTPUTS:
        dates (list) - dates, in the order of url_list
        """
        #make sure the input is of the correct type
        if type(url_list) is str:
            url_list = [url_list]
        if type(url_list) is not list:
            print("This is a {} and needs to be a list:\n{}").format(type(url_list), url_list)

        # go through the url list, get the html from the url, and find the date
        dates = []
        with open(self.date_file, 'a') as f:
            writer = csv.writer(f)
            for i, url in enumerate(url_list):
                post = _gen_submission(url)
                post_date = post.created_utc
                dates.append(post_date)
                writer.writerow(url, post_date)
                if rand_time:
                    time.sleep(time_delay+random.randint(0,5))
                else:
                    time.sleep(time_delay)

        return dates

    """
    def set_praw_params(self, *kwargs):
        valid_kwargs = {'client_id':client_id,
                     'client_secret':client_secret,
                     'password':password,
                     'user_agent':'testscript by /u/fakebot3',
                     'username':username}
        for kwarg in *kwargs:
    """


    def _query_site(self, url):
        """
        Retrieves the html content from a single url
        Exits if it receives an invalid code, with the code received
        INPUTS:
        url (string) - url to visit
        OUTPUTS:
        res.text (string) - the html content from the url
        """
        res = requests.get(url)
        if not res.status_code == requests.codes.ok:
            print("WARNING: status code", res.status_code)
            print('\n')
        else:
            print("Loading html from: ")
            print(url)
            return res.text

    def _parse_html(self, html_str):
        """
        Parses html to find the date of the post
        INPUTS:
        html_str (string) - html from a page
        OUTPUTS:
        date -
        """
        #TODO finish documentation when you figure out date type
        soup = BeautifulSoup(html_str, 'html.parser')
        date = soup.select("p[class='tagline'] > time")
        #TODO: need to further refine date to just day
        return date

    def scrape_dates(self, url_list, time_delay=5, rand_time=True):
        """
        Runs _query_site and _parse_html for every url in url_list, returning
        the dates in the order of the urls.
        INPUTS:
        url_list (list of strings) - urls to scrape
        time_delay (int) - how long to wait in between urls (in seconds). Defaults to 5 seconds.
        OUTPUTS:
        dates (list) - dates, in the order of url_list
        """
        #TODO need to make sure the wait time is correct before using
        #in case url is a single string
        if type(url_list) is str:
            url_list = [url_list]
        if type(url_list) is not list:
            print("This is a {} and needs to be a list:\n{}").format(type(url_list), url_list)
        # go through the url list, get the html from the url, and find the date
        dates = []
        for i, url in enumerate(url_list):
            html_text = _query_site(url)
            print("{}/{}/n").format(i, len(url_list))
            date = _parse_html(html_text)
            #TODO need to further refine date (Either here or in above fn)
            dates.append(date)
            #be sure to wait before repeating to avoid getting banned
            if rand_time:
                time.sleep(time_delay+random.randint(0,10))
            else:
                time.sleep(time_delay)

        return dates



"""
Note:
A potential snag: the coarse-discourse dataset doesn't take date into account.
There is  no date information in any column, and it is not ordered according to date.

If I want to download only a small subset of the reddit data (for instance,
1 month), I will need to scrape reddit (or use its API), using the url in my
dataframe to find the date.

Note: On comments metadata
ids are fullnames: type+id
types: t1_=comment, t2_=account, t3_=link, t4_=message, t5_=subreddit, t6=_award
"""


if __name__ == '__main__':
    # get the list to test on
    coarse_json = 'data/coarse_discourse_dataset.json'
    coarse_df = load_discourse_data(coarse_json)
    test_urls = list(coarse_df.url[:10])
    # try to scrape them
    Date_Finder = Reddit_Scraper(date_file='data/list_of_dates.csv', error_file='data/error_log.csv')
    Date_Finder.login(user_agent='testing date_finder by ArSlatehorn')
    dates = Date_Finder.get_dates(url_list)
