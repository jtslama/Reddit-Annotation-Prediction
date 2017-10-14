import pandas as pd
import json
import ast
import requests
from bs4 import BeautifulSoup


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
    pass

"""
Note:
A potential snag: the coarse-discourse dataset doesn't take date into account.
There is  no date information in any column, and it is not ordered according to date.

If I want to download only a small subset of the reddit data (for instance,
1 month), I will need to scrape reddit (or use its API), using the url in my
dataframe to find the date.
"""

class date_finder(object):
    """
    To solve a problem I didn't know I had
    """
    def __init__(self, url_list):
        self.url_list = urls

    def query_site(self, url):
        res = requests.get(url)
        if not res.status_code == res.codes.ok:
            print "WARNING: status code ", res.status_code
        else:
            return res.text

    def parse_html(self, html_str):
        soup = BeautifulSoup(html_str, 'html.parser')
        date = soup.select("p[class='tagline'] > time")



if __name__ == '__main__':
    orig_json_file = 'coarse-discourse/coarse_discourse_dataset.json'
    df = load_discourse_data(orig_json_file)
    df.url[0]
    df.posts[0]
    df.head(1)
    len(df)

    res = requests.get(url)
    if not res.status_code == res.codes.ok:
        print "WARNING: status code ", res.status_code
    else:
        return res.text


    soup = BeautifulSoup(html_str, 'html.parser')
    date = soup.select("p[class='tagline'] > time")
