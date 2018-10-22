from lda2vec import preprocess, Corpus
import numpy as np
import pandas as pd
import logging
import pickle
import os.path
import json
logging.basicConfig()

max_length = 10000   # Limit of 250 words per comment
min_author_comments = 1  # Exclude authors with fewer comments
nrows = None  # Number of rows of file to read; None reads in full file
# Preprocessing and save preprocessed data to npy file.
def preprocessing_save():
    json_filename = 'data/NoDAPLsmall.json'
    json_file = open(json_filename)
    json_str = json_file.read()
    json_data = json.loads(json_str)
    sm_corpus = list()
    for record in json_data:
        sm_corpus.append(record['Sentences'])
    tokens, vocab = preprocess.tokenize(sm_corpus, max_length, n_threads=4,merge=False)
    del sm_corpus
    np.save("preprocessed/tokens.npy",tokens)
    np.save("preprocessed/vocab.npy",vocab)

if __name__=='__main__':
    # preprocessing_save()
    tokens = np.load("preprocessed/tokens.npy.npy")
    vocab = np.load("preprocessed/vocab.npy.npy")
    # Make a ranked list of rare vs frequent words
    corpus = Corpus()
    corpus.update_word_count(tokens)
    corpus.finalize()
    # The tokenization uses spaCy indices, and so may have gaps
    # between indices for words that aren't present in our dataset.
    # This builds a new compact index
    compact = corpus.to_compact(tokens)
    # Remove extremely rare words
    pruned = corpus.filter_count(compact, min_count=10)
    # Words tend to have power law frequency, so selectively
    # downsample the most prevalent words
    clean = corpus.subsample_frequent(pruned)
    print "n_words", np.unique(clean).max()

    # # Extract numpy arrays over the fields we want covered by topics
    # # Convert to categorical variables
    # author_counts = features['comment_author'].value_counts()
    # to_remove = author_counts[author_counts < min_author_comments].index
    # mask = features['comment_author'].isin(to_remove).values
    # author_name = features['comment_author'].values.copy()
    # author_name[mask] = 'infrequent_author'
    # features['comment_author'] = author_name
    # authors = pd.Categorical(features['comment_author'])
    # author_id = authors.codes
    # author_name = authors.categories
    # story_id = pd.Categorical(features['story_id']).codes
    # # Chop timestamps into days
    # story_time = pd.to_datetime(features['story_time'], unit='s')
    # days_since = (story_time - story_time.min()) / pd.Timedelta('1 day')
    # time_id = days_since.astype('int32')
    # features['story_id_codes'] = story_id
    # features['author_id_codes'] = story_id
    # features['time_id_codes'] = time_id

    # print "n_authors", author_id.max()
    # print "n_stories", story_id.max()
    # print "n_times", time_id.max()

    # # Extract outcome supervised features
    # ranking = features['comment_ranking'].values
    # score = features['story_comment_count'].values

    # # Now flatten a 2D array of document per row and word position
    # # per column to a 1D array of words. This will also remove skips
    # # and OoV words
    # feature_arrs = (story_id, author_id, time_id, ranking, score)
    # flattened, features_flat = corpus.compact_to_flat(pruned, *feature_arrs)
    # # Flattened feature arrays
    # (story_id_f, author_id_f, time_id_f, ranking_f, score_f) = features_flat

    # # Save the data
    # pickle.dump(corpus, open('corpus', 'w'), protocol=2)
    # pickle.dump(vocab, open('vocab', 'w'), protocol=2)
    # features.to_pickle('features.pd')
    # data = dict(flattened=flattened, story_id=story_id_f, author_id=author_id_f,
    #             time_id=time_id_f, ranking=ranking_f, score=score_f,
    #             author_name=author_name, author_index=author_id)
    # np.savez('data', **data)
    # np.save(open('tokens', 'w'), tokens)