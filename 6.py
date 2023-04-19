from os import listdir
import os
from os.path import isfile, join
import re
from nltk.stem import PorterStemmer
from functools import lru_cache

import numpy as np


def load_text(path: str):
    '''Load text from file'''

    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def delete_punctuation(text: str):
    '''Delete punctuation from text'''

    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text.strip()


def filter_text(text: str):
    '''Filter text by using stop list'''

    stop_list = ["a", "an", "and", "the", "to", "in", "of", "for", "on", "at",
                 "with", "by", "from", "up", "about", "into", "over", "after", 
                 "between", "out", "through", "during", "under", "before", "above", 
                 "below", "since", "without", "within", "along", "except", "among", 
                 "beyond", "toward", "until", "upon", "regarding", "amongst", "per", 
                 "throughout", "towards", "versus", "via", "whether"]
   
    text = ' '.join([word for word in text.split() if word not in stop_list])
    return text



def stem_sentences(text):
    '''Stem sentences'''

    ps = PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


@lru_cache
def get_index():
    '''Get index for all files in gutenberg folder'''

    target_folder = 'gutenberg'
    paths = [path for path in listdir(target_folder) if isfile(join(target_folder, path))]
    index = {}
    for path in paths:
        text = f' {load_text(join(target_folder, path))}'
        text = delete_punctuation(text)
        text = filter_text(text)
        text = stem_sentences(text)
        for word in text.split():
            word = word.strip()
            if word not in index:
                index[word] = set()
            index[word].add(path)
    return index, paths


def get_index_prepped(path):
    dir_list = os.listdir(path)
    file_indices = {}
    for file in dir_list:
        file_indices[file] = dir_list.index(file)

    documents_dict = {}
    for file in dir_list:
        index = file_indices[file]
        with open(path+os.sep+file, "r", encoding='utf8') as f:
            text = f.read()
        f.close()
        for word in text.split():
            if word not in documents_dict:
                documents_dict[word] = set()
            documents_dict[word].add(index)


    return file_indices, documents_dict


def query(index, querystr):
    '''Search in index'''

    find = querystr.split('AND')
    all_words = ' '.join(find)
    all_words = delete_punctuation(all_words)
    all_words = filter_text(all_words)
    all_words = stem_sentences(all_words)
    find = all_words.split()

    found_in = set()
    for word in find:
        word = word.strip()
        if word in index:
            if found_in:
                found_in = found_in.intersection(index[word])
            else:
                found_in = index[word]
    return found_in


def compute_term_freq(path, file_indices):
    dir_list = os.listdir(path)
    frequencies_dict = {}
    for file in dir_list:
        index = file_indices[file]
        with open(path+os.sep+file, "r", encoding='utf8') as f:
            text = f.read()
        for word in text.split():
            if word not in frequencies_dict:
                frequencies_dict[word] = {}
            if index not in frequencies_dict[word]:
                frequencies_dict[word][index] = 0
            frequencies_dict[word][index] += 1

    return frequencies_dict


def compute_inv_doc_freq(file_indices, index):
    inv_doc_frequencies = {}
    N = len(file_indices)
    for word in index:
        df_t = len(index[word])
        inv_doc_frequencies[word] = np.log(N/df_t)
    return inv_doc_frequencies


def compute_tfidf(term_freq, inv_doc_freq):
    tfidf = {}
    for word in term_freq:
        tfidf[word] = {}
        for file in term_freq[word]:
            tfidf[word][file] = term_freq[word][file] * inv_doc_freq[word]
    return tfidf


def score(tfidf, file_indices, query):
    scores = {}
    for file in file_indices.values():
        scores[file] = 0
    find = query.split('AND')
    all_words = ' '.join(find)
    all_words = delete_punctuation(all_words)
    all_words = filter_text(all_words)
    all_words = stem_sentences(all_words)
    find = all_words.split()
    for word in find:
        if word in tfidf:
            for file in tfidf[word]:
                if file in tfidf[word]:
                    scores[file] += tfidf[word][file]
    return scores


def printTopScores(scores, file_indices, n):
    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_scores = sorted_scores[:n]
    for score in top_scores:
        print("Score: " + str(score[1]) + ", File: " + list(file_indices.keys())[list(file_indices.values()).index(score[0])] + ", Index: " + str(score[0]))



def main():
    '''Main function'''
    targetFolder = "gutenberg_stemmed"
    path = targetFolder

    paths, index = get_index_prepped(path)

    term_freq = compute_term_freq(path, paths)
    inv_doc_freq = compute_inv_doc_freq(paths, index)
    tfidf = compute_tfidf(term_freq, inv_doc_freq)

    query_str = "god"
    print(f'evaluating query: {query_str}')
    scores = score(tfidf, paths, query_str)
    print(query(index, query_str))
    printTopScores(scores, paths, 10)


if __name__ == "__main__":
    main()
