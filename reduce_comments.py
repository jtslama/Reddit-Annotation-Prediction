import pandas as pd
import datetime
import time
import sys
import json

class CommentFinder(object):
    """
    In the case you want to use python locally to search a subset of the data
    for ids that match ids used in the annotations.
    """
    def __init__(self, filelist, write=True):
        self.filelist = filelist
        self.write = write

    def load_data_from_jsons(self, filepath):
        """
        The steps necessary to take the file of a list of json objects and compile it
        into a pandas DataFrame
        INPUTS:
        filepath (string) - file path or name of file (made up of list of json
                            objects)
        OUTPUTS:
        df (pandas dataframe) - dataframe of the objects in filepath
        """
        # read file into a list
        START = time.time()
        with open(filepath, 'r') as f:
            rows = f.readlines()
        T1 = time.time()
        print "reading {} took {}s, is {}".format(filepath, round(T1-START, 2), self._byte_to_larger(sys.getsizeof(rows) ))
        # remove newline chars
        rows = [r.rstrip() for r in rows]
        # now each string can be made to a dictionary
        dictified = [json.loads(r) for r in rows]
        T3 = time.time()
        print "dictifying took {}s, is {}".format(round(T3-T1, 2), self._byte_to_larger(sys.getsizeof(dictified)))
        # then loaded into a pandas DataFrame
        df = pd.DataFrame(dictified)
        T4 = time.time()
        print "df-ify took {}s, is {}".format(round(T4-T3, 2), self._byte_to_larger(sys.getsizeof(df)))
        return df

    def _byte_to_larger(self, bytes, order=['KB', 'MB', 'GB', 'TB'], current = 'B'):
        "Quick and dirty bytes converter to make write_stats look cleaner."
        next_num = bytes/1024.0
        if next_num < 1.0:
            return "{:.2f} {}".format(bytes, current)
        else:
            return self._byte_to_larger(next_num, order[1:], order[0])

    def _create_csv(self, filename):
        df = load_data_from_jsons(filename)
        df.to_csv(filename+'_translated.csv')

    def find_annotated_comments(self, annotations, comments):
        """finds the intersection of annotatoins and comments where both contain
        features relating to individual comments. They match on annotations.id and
        comments.name (both pandas dfs)
        """
        # create set of annotation ids
        annotations.set_index('id', inplace=True)
        comments.set_index('name', inplace=True)
        results = annotations.join(comments, how='left', lsuffix='_ann', rsuffix='_comm')

        return results


    def search_files(self, annotations):
        #time used for debugging
        searched = {}
        results = []
        for f in self.filelist:
            START1 = time.time()
            print("loading file {} into dataframe...".format(f))
            comment_df = self.load_data_from_jsons(f)
            END1 = time.time()
            print("{} loaded, took {}m".format(f, round((END1-START1)/60, 2)))

            df = self.find_annotated_comments(annotations, comment_df)

            if self.write:
                T1 = time.time()
                df.to_csv('{}_searched.csv'.format(f))
                T2 = time.time()
                print("{} written to csv, took {}m".format(f, round((T2-T1)/60, 2)))

            searched[f] = df
            results.append(df)
            END2 = time.time()
            print("{} searched, took {}min".format(f, round((END2-END1)/60, 2)))

        final_results = pd.concat(results)
        if self.write:
            T1 = time.time()
            final_results.to_csv("results_composite.csv")
            T2 = time.time()
            print("composite written to csv, took {}m".format(round((T2-T1)/60, 2)))

        return final_results


if __name__ == '__main__':
    #use time module for debugging purposes, here and in the class above
    START = time.time()

    annotations = pd.read_csv('annotations_table.csv')
    T1 = time.time()
    print("annotations loaded, took {}s".format(round(T1-START), 2))

    #run for a small file (local)
    filelist = ['data/raw/RC_2008-11']

    #or run for a bigger file set (for EC2 instance)
    # filelist = ["zips/R_data/RC_2016-04", "zips/R_data/RC_2016-05"]
    Search = CommentFinder(filelist)

    Search.search_files(annotations)

    T2 = time.time()
    print("search complete, took {}m".format(round((T2-START)/60, 2)))
