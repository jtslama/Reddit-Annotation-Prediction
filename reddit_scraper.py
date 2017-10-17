import requests
from bs4 import BeautifulSoup
import time
import random
import csv
import praw
from initial_data_loading import load_discourse_data
from reddit_config import login_details


class Reddit_Scraper(object):
    """
    Designed to use the Reddit API to find the submission dates for a long list
    of posts
    INPUTS:
    date_file (string) - filename of csv where dates and urls where be written
                         to (as date url pairs)
    """
    def __init__(self, login_details, date_file='list_of_dates', error_file='error_log'):
        #TODO: figure out how to setup the login stuff
        """
        Sets up default files for results and errors, creates praw instance for
        scraping and feeds it user information to log in.
        INPUTS:
        login_details (dict) - a dictionary containing the client key-value pairs
        date_file (string) - filename where dates will be written (optional)
        error_file (string) - filename where errors will be written (optional)
        OUTPUTS:
        none
        """
        # set up files to be written to
        self.date_file = date_file
        self.error_file = error_file
        # create praw instance, log in
        self.reddit = praw.Reddit(client_id=login_details['CLIENT_ID'],
                                  client_secret=login_details['CLIENT_SECRET'],
                                  password=login_details['PASSWORD'],
                                  user_agent=user_agent,
                                  username=login_details['USERNAME'])

    def _get_submission(self, url):
        #TODO: figure out try-except block standard practice
        """
        Attempts to retrieve a submission object from a url, logs an error
        if it cannot.
        """
        with open(self.error_file, 'a') as f:
            try:
                req = self.reddit.submission(url=url)

            except:
                writer = csv.writer(f)
                writer.writerow(["failed to return {url}".format(url)])
        return req

    def _get_date(self, submission, url):
        #TODO: figure out try except block standard practice
        """
        Attempts to get the date created attribute (created_utc, in epoch
        seconds) from a submission object, logs an error if it cannot.
        """
        with open(self.error_file, 'a') as f:
            try:
                date = submission.created_utc
                return date
            except HTTPError:
                error =  submission.response.status_code
                print("Error: {}".format(error))
                writer = csv.writer(f)
                writer.writerow(["failed to return date from {url} with error {}"
                                .format(url, error)])
                return None
            finally:
                print("ran into an error on submission")
                return None



    def get_dates(self, url_list, time_delay=1.25, rand_time=False):
        """
        Finds the date a submission was posted from a list of submission urls
        INPUTS:
        url_list (list of strings) - urls to scrape the submission dates from
        time_delay (int) - how long to wait in between urls (in seconds).
                           Defaults to 1.25 seconds. Reddit API is max 60
                           reqs/min.
        rand_time - whether or not to randomize the time in between requests
        OUTPUTS:
        dates (list) - dates, in the order of url_list
        """
        # make sure the input is of the correct type
        if type(url_list) is str:
            url_list = [url_list]
        if type(url_list) is not list:
            print("This is a {} and needs to be a list:\n{}")
                 .format(type(url_list), url_list)

        # go through the url list, get the html from the url, and find the date
        # write the date to the date_list, and append it to dates to be returned
        dates = []
        with open(self.date_file, 'a') as f:
            writer = csv.writer(f)
            for i, url in enumerate(url_list):
                print("{}/{} : {}".format(i+1, len(url_list), url))
                post = self._get_submission(url)
                post_date = self._get_date(post, url)
                dates.append(post_date)
                writer.writerow([post_date, url])
                if rand_time:
                    time.sleep(max(1, 0.5+random.gauss(1, 1))
                else:
                    time.sleep(time_delay)
        return dates


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
        date (string) - post date
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
        time_delay (int) - how long to wait in between urls (in seconds).
                           Defaults to 5 seconds.
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
            print("{}/{}").format(i, len(url_list))
            date = _parse_html(html_text)
            #TODO need to further refine date (Either here or in above fn)
            dates.append(date)
            #be sure to wait before repeating to avoid getting banned
            if rand_time:
                time.sleep(time_delay+random.randint(0, 10))
            else:
                time.sleep(time_delay)

        return dates


if __name__ == '__main__':

    # get the list to test on
    coarse_json = 'data/coarse_discourse_dataset.json'
    print("Loading {}...".format(coarse_json))
    coarse_df = load_discourse_data(coarse_json)
    urls = list(coarse_df.url)
    # scraper broke on url #1983
    urls = urls[1982:]
    print("{} loaded".format(coarse_json))
    # try to scrape them
    Date_Finder = Reddit_Scraper(date_file='data/list_of_dates.csv',
                                 error_file='data/error_log.csv')
    print("logging in...")
    Date_Finder.login(login_details,
                      user_agent='testing date_finder by ArSlatehorn')
    print("searching for dates...")
    dates = Date_Finder.get_dates(urls)
