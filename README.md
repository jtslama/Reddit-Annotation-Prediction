## Reddit Annotation Classification
by John Slama, 10/9/17-10/30/17

##### Categorizing discourse on Reddit, with a focus on identifying negative comments

![datascope](app/static/images/datascope.PNG)


## Overview
Reddit is one of the most popular social media sites on the web. In October of 2017, an average of 5 million comments were made each day across its network of 50,000 active subreddits, which serve as internal communities. Each of these communities is moderated by a team of volunteers, who rely on static filters and user reports to filter out comments which break their rules. I designed a model which could quickly identify the kind of comments most communities would want to eliminate.


## Data Source
This project was inspired by the [Google Coarse Discourse Dataset](http://github.com/google-research-datasets/coarse-discourse). The folks at Google were kind enough to have a team of annotators go through a random sampling of Reddit threads and classify the comments into 10 different types: Question, Answer, Appreciation, Elaboration, Agreement, Disagreement, Other, None, Humor and Negative Reaction. I re-constructed their dataset (the comment bodies), along with some additional metadata (such as the post score), and used it to train an ensemble which could identify the Negative Reactions. These comments are responses to a previous comment which do not discuss the merits of a prior comment or offer a correction, but instead attack or mock the commenter.

## Data Scope
The researchers who created the Coarse Discourse dataset did so by randomly sampling 50,000 threads (discussion prompts) from Reddit created between its inception and May 2016 (approximately 238 million threads). They had teams of 3 annotators label the top 40 comments from each post according to the criteria they set for each label (more detail can be found here, under Discourse Act Definitions).
#TODO: link here
The dataset does not have any information about the comment contents or any metadata other than the thread url, which means that the dataset must be reconstructed before it can be of use.

## Dataset Reconstruction

1) Determine what data I need to acquire, and what I can leave alone. Reddit has amassed billions of comments, and I need no more than 200,000.Reddit comments have been archived by several enthusiasts (sources), and are stored in month-long segments.
* Build custom scraper to determine which months the sampled urls were created.
* Download comment data for months the posts were created in and the following months (to address late comments and end of month posts) to an Amazon EC2 instance (total of 1.5TB of data)

2) Filter out all the comments and threads I don't need.
* Select only comments that are in annotated threads.
* Spin up an Amazon Elastic MapReduce cluster to parallelize this filter. Doing this locally would have taken an estimated 3 weeks. Doing this with EMR took under an hour.

## Preparation

I built a clean-tokenize-TFIDF-SVD pipeline to convert the comments to something I could train a model on.
1. Clean - Remove extraneous metadata and labels, deal with anything which might cause the model to crash.
2. Tokenize - Convert the comment bodies into a list of separate stemmed words (running and runs are different words, but share a stem), removing punctuation and stopwords (e.g. 'a', 'the').
3. [TFIDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf) - Compile a vocabulary of stemmed words in the corpus, use it to build a Document-Term feature matrix ('X') from the most important terms.
4. [Single Value Decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition) - The Document-Term matrix contains over 150,000 columns. Singular Value Decomposition allows me to determine which vocabulary is most important. 2000 features account for over 90% of the feature importance, so I restrict to those.

My final dataset used 2000 vocabulary features and 7 features related to the comment meta-data (such as the score of the comment, whether it was gilded, and so on). The training dataset contained approximately 80,000 comments. To train on such a large dataset, I used my Amazon EC2 instance (m4xlarge).

## Initial Modeling
The data is skewed: approximately 40% of the comments are Answers and less than 2% are Negative Reactions. As a result, basic classifiers (Random Forest, Naive Bayes, Logistic Regression, etc.) result in high accuracy (most things aren't Negative Reactions) and very low recall and precision. This isn't far different from a baseline model that just makes a random guess in proportion to the label distribution (i.e. it guesses Negative Reaction approximately 2% of the time), which performs similarly, though it is better than a model which just guesses the most common class (nothing is a Negative Reaction).

![BResults](images/BasicResults.PNG)

## Ensemble
To address this issue, I created an ensemble model. I set up 3 tiers.

![EnsembleStructure](images/EnsembleStructure.PNG)

* 1) Tier 1 is a model which determines whether the comments are Answers/Elaborations or something else (all other categories). If it predicts they are not Answers or Elaborations, it passes them to the second tier.
* 2) Tier 2 determines whether the comments are Humor/Negative Reactions or something else (Question, Disagreement, Agreement, Appreciation). It passes the Humor and Negative Reactions to the final tier.
* 3) Tier 3 makes the final determination of whether a comment is a Negative Reaction or Humor.

Each tier can have a different model type that can be tuned with different hyperparameters. The ensemble setup I used consisted of a GradientBoostingClassifier for Tier 1, an AdaBoostClassifier for Tier 2, and a GradientBoostingClassifier for Tier 3.

![FResults](images/FResults.PNG)

## Results
The Ensemble greatly outperforms the baseline model and the basic classifiers. The metric of importance here is precision, the fraction of declared postives (classified as Negative Reactions) that are true positives (actual Negative reactions). The higher this measure is, the better this tool is. We want an aid that will highlight comments for additional moderation. To avoid wasting the moderators' time, we want to avoid false positives (falsely classifying a comment as a Negative Reaction when it is not), and precision tells us how well we are doing. Secondary to this is Recall, how many of the Negative Reactions our model is catching. In both metrics, the Ensemble greatly outperforms the baselines and basic models.

## Future Work
There is some information that I feel is not being captured by the features I've chosen. In addition to vocabulary, I could look for phrases (n-grams). I could also include information about a parent comment (either its own contents, or some metadata such as how far apart in time the two comments were made). I could also do some categorizing by subreddit (which community the discussion is taking place in has a large effect on the social norms of discourse). Finally, I'd like to experiment with different ensemble makeups - for instance, 1 tier per label type. Time constraints mean that this project still has unrealized potential that might be of interest.

## Credits:
[Google's Coarse Discourse Dataset](http://github.com/google-research-datasets/coarse-discourse) - The group that did the legwork. They had this idea first, and they paid people to read and annotate reddit. Their paper has much more detail than is included here.

[Reddit API](https://www.reddit.com/dev/api) - makes getting information about thread metadata much easier and faster.

[Amazon Web Services](https://aws.amazon.com/) - computing power and memory on demand. This is what made it possible for me to store than my machine could handle, and to sift through it in a reasonable amount of time.

[BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/) - A python web-scraping library. It greatly simplifies finding particular elements in a complex webpage.

[Natural Language Toolkit (nltk)](http://www.nltk.org/) - Provides support for natural language processing, including stopwords lists, and word tokenizers.

[scikit-learn](http://scikit-learn.org/stable/) - An indispensible machine learning library that contains the code and documentation for the individual classifiers.

[Galvanize](https://www.galvanize.com) - The Data Science school.


[More about Reddit](https://expandedramblings.com/index.php/reddit-stats/)
