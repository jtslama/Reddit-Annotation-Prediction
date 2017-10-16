import pandas as pd
import json
import ast
import requests
from bs4 import BeautifulSoup
import time


def load_discourse_data(filepath):
    """
    Since I keep doing it, and its a pain every time:
    The steps necessary to take the file of a list of json objects and compile it
    into a pandas DataFrame
    """
    # read file into a list
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # remove newline chars, change true (invalid) to True (bool)
    rows = [lin.rstrip().replace('true', 'True') for lin in lines]
    rows_fixed = [r.replace('true', 'True') for r in rows]
    # now each string can be made to a dictionary
    dictified = [ast.literal_eval(rf) for rf in rows_fixed]
    # then loaded into a pandas DataFrame
    df = pd.DataFrame(dictified)
    return df




def load_reddit_data(filepath):
    """
    Since I keep doing it, and its a pain every time:
    The steps necessary to take the file of a list of json objects and compile it
    into a pandas DataFrame
    """
    filepath = 'data/RC_2008-11'
    # read file into a list
    with open(filepath, 'r') as f:
        rows = f.readlines()
    # remove newline chars, change true (invalid) to True (bool), false to False, null to None
    switch = {'true': 'True',
            'false': 'False',
            'null': 'None'}
    for item in switch:
        rows = [r.replace(item, switch[item]) for r in rows]
    rows = [r.rstrip() for r in rows]
    # now each string can be made to a dictionary
    dictified = [ast.literal_eval(rf) for rf in rows]
    # then loaded into a pandas DataFrame
    df = pd.DataFrame(dictified)
    return df
sample_reddit_data = 'data/RC_2008-11'
df = load_reddit_data(sample_reddit_data)
df.columns
df[df.author == 'ironpony'].head(1)
len(df.subreddit_id.unique())



%cd /Users/jt/Desktop/Galvanize/DSI_Capstone/
filepath = 'data/RC_2008-11'
df = load_discourse_data(filepath)
with open(filepath, 'r') as f:
    rows = f.readlines()
type(rows[0])
rows[0]

"""
Note:
A potential snag: the coarse-discourse dataset doesn't take date into account.
There is  no date information in any column, and it is not ordered according to date.

If I want to download only a small subset of the reddit data (for instance,
1 month), I will need to scrape reddit (or use its API), using the url in my
dataframe to find the date.
"""

class Reddit_Post_Date_Finder(object):
    """
    To solve a problem I didn't know I had
    TODO: datetime handling, what goes in init?
    """
    def __init__(self):
        pass

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

    def get_dates(self, url_list, time_delay=5):
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
            time.sleep(time_delay)

        return dates





if __name__ == '__main__':
    orig_json_file = 'data/coarse_discourse_dataset.json'
    df = load_discourse_data(orig_json_file)
    df.head()
    len(df)

    # looking at a small sample of reddit data


    # small test case (to test scraper)
    urls = df.url.tolist()[:10]
    Finder = Reddit_Post_Date_Finder()
    dates = Finder.get_dates(urls, time_delay=10)
