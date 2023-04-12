from os import listdir
from os.path import isfile, join
import re
from nltk.stem import PorterStemmer
from functools import lru_cache


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
    return index

def write_text(path: str, text: str):
    '''Load text from file'''

    with open(path, 'w+', encoding='utf-8') as file:
        file.write(text)

def write_index():
    '''Get index for all files in gutenberg folder'''

    target_folder = 'gutenberg'
    target_folder_stemmed = 'gutenberg_stemmed'
    paths = [path for path in listdir(target_folder) if isfile(join(target_folder, path))]

    for path in paths:
        text = f' {load_text(join(target_folder, path))}'
        text = delete_punctuation(text)
        text = filter_text(text)
        text = stem_sentences(text)
        write_text(join(target_folder_stemmed, path), text)


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


def main():
    '''Main function'''

    write_index()
    #index = get_index()
    #print(paths)
    #find = 'bellow AND small'
    #found_in = query(index, find)
    #print(found_in)


if __name__ == "__main__":
    main()
