# Problem Statement
Two subreddits were chosen, r/nosleep and r/HFY.
- r/nosleep is a subreddit for horror/supernatural related short stories.
- r/HFY is a subreddit for science fiction/fantasy related short stories.

Both subreddits are short stories subreddits, but of different genres.

However, the 2 genres had some overlapping themes (i.e. some short stories in r/HFY had horror elements to it, and vice versa).
Using natural language processing, models were constructed to see if the two subreddits could be classified or not.


# Executive Summary
- Around 1,000 posts from each of the chosen subreddits were scraped from reddit using the reddit API.

- These were stored in the respectively named .csv files, and then loaded into dataframes. The 2 dataframes were then combined and preprocessing/cleaning was done on the scraped posts (using techniques such as RegEx, tokenizing and lemmatization).

- As this is a binary classification problem, Multi-nomial Naive-Bayes and Logistic Regression were chosen as the 2 models to use. For each model, 2 different word embedding methods were used (count vectorization and TFIDF). The whole modelling process was done using pipelines and then hyperparameters were optimized using GridSearchCV.

- It was found that all models constructed performed well, with the Logistic regression model utilizing count vectorization as the word embedding method performing the best (but not significantly better).

- Confusion matrices were constructed for each model, however it was discussed that in the context of this project accuracy score is the most important metric.


# Data Dictionary
- Data dictionary for the features used in the models:
|Feature|Data type|Description|
|-------|---------|-----------|
|selftext|*str*|The content of the reddit post (the predictor variable)|
|nosleep|*int*|Whether the post is from r/nosleep (1) or not (0) --- (the target variable)|


# Conclusion
- The selected model (nb_gs1_pkl) which uses count vectorization and a naive-bayes model achieved consistently good accuracy scores.
    - Suggests the model does not overfit to the training data and can be generalised to real-world unseen data.
    - For any posters to r/nosleep or r/HFY, there is a high degree of reliability (~ 95%) that the model can correctly classify into which subreddit the post belongs, in case the posts have initially been misclassified, even in the absence of any titles/words giving away the subreddit.
- Overfitting seemed to be an issue when it came to modelling with Logistic Regression.
    - A comparison between Naive-Bayes and Logistic Regression was given, with a possible reason for the overfitting attributed to insufficient number of samples.


- **Limitations**:
    - For subreddits with highly similar topics, the accuracy will likely be too low to make the constructed model useful in practical scenarios.
        - i.e., the models constructed in this project probably cannot pick up on nuances and context that a typical human can use to discern between 2 similar topics.
    - The best model chosen in this project (Naive-Bayes with count vectorization) is appropriate for problems with relatively small sample size.
        - However, if sample size were to tend to infinity (i.e. very large datasets), Logistic Regression might be a better choice assuming the problem is still a binary classification problem.
        - This is due to the inherent erroneous assumption for Naive-Bayes, that all features are independent of each other.
    - The 'bag-of-words' methodology used in this project creates sparse matrices from word embedding methods (count vectorization & tf-idf vectorization)
        - Sparse matrices represent a problem for other classification models such as KNN, since there are no distinct clustering of features.
        
        
- **Possible further work**:
    - Move outside the 'bag-of-words' (BoW) methodology as context is not considered at all.
        - Explore effectiveness of Neural Networks as a possible (better) alternative to BoW.
    - Expand this problem from a binary classification problem to a multi-class classification problem (i.e. classifying posts that could belong to any of more than 2 subreddits)
        - This may prove difficult, but is a lot more practical in real world usage. 