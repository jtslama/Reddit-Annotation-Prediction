"""
A few functions designed to create a file with a list of urls to download.
Designed for primary data acquisition only (not made to be modular or reused)
"""


def write_url_list(filename, base_url=None):
    """
    Designed create a text file full of the urls where the reddit comment
    dataset is stored
    INPUTS:
    filename (string): the desired name of the file which the urls will be
                       added to (can create a file if none exists)
    base_url (string): on the off-chance this script needs to be run on a
                       different website (should never occur)
    OUTPUTS:
    creates a file in the working directory of name <filename>
    """
    # uses standard base_url if none has been entered
    if not base_url:
        base_url = 'https://archive.org/download/2015_reddit_comments_corpus/reddit_data'

    # creates a url string for each month of each year in xrange (or months
    # 10-12 in 2007)
    urls = []
    for n in xrange(2007, 2016):
        if n == 2007:
            for i in xrange(10, 13):
                url_path = "{}/{}/RC_{}-{}.bz2".format(base_url, n, n, i)
                urls.append(url_path)
        else:
            for i in xrange(1, 10):
                url_path = "{}/{}/RC_{}-0{}.bz2".format(base_url, n, n, i)
                urls.append(url_path)
            for i in xrange(10, 13):
                url_path = "{}/{}/RC_{}-{}.bz2".format(base_url, n, n, i)
                urls.append(url_path)

    # writes all the urls to a file of name <filename>
    with open(filename, 'w+') as f:
            f.write("\n".join(urls))
    return urls


def makeup_url_list(filename):
    """
    Earlier program missed some urls. This fills in the gaps
    INPUT: filename - string of filename to write to
    OUTPUT: creates or modifies filename with list of urls
    """

    base_url = 'https://archive.org/download/2015_reddit_comments_corpus/reddit_data'
    urls = []
    for n in xrange(2008, 2016):
        for i in xrange(1, 10):
            url_path = "{}/{}/RC_{}-0{}.bz2".format(base_url, n, n, i)
            urls.append(url_path)
    with open(filename, 'w+') as f:
        f.write("\n".join(urls))
    return urls


if __name__ == '__main__':
    # ~150GB to download
    # make full url list, write file
    """NOTE: Terminated at RC_2013-10 (<-needs to be downloaded)"""
    urls = makeup_url_list()
    # use a subsection for test purposes
    test_url = urls[0]
