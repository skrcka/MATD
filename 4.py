from os import listdir
from os.path import isfile, join
import re
from nltk.stem import PorterStemmer


def load_text(path: str):
    '''Load text from file'''

    with open(path, 'r') as file:
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


def main():
    '''Main function'''

    target_folder = 'gutenberg'
    paths = [path for path in listdir(target_folder) if isfile(join(target_folder, path))]
    text = ''
    for path in paths:
        text += f' {load_text(join(target_folder, path))}'
    text = delete_punctuation(text)
    print(f'Text without punctuation: {text[:100]}')
    text = filter_text(text)
    print(f'Filtered text: {text[:100]}')
    text = stem_sentences(text)
    print(f'Stemed text: {text[:100]}')


if __name__ == "__main__":
    main()
