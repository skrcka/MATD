import random
import time


def brute_serach(string, substring):
    occurences = []
    for i in range(len(string) - len(substring) + 1):
        j = 0
        while j < len(substring) and string[i + j] == substring[j]:
            j += 1
        if j == len(substring):
            occurences.append(i)
    return occurences

def substring_search(string, substring):
    transitions = {}
    for i in range(len(substring)):
        for c in set(string + substring):
            if i < len(substring) and c == substring[i]:
                transitions[(i, c)] = i + 1
            else:
                k = min(i, len(substring))
                s = substring[:k] + c
                while substring[:k] not in s[:-1] and k > 0:
                    k -= 1
                    s = substring[:k] + c
                transitions[(i, c)] = k
        
    state, occurrences = 0, []
    for i, c in enumerate(string):
        state = transitions.get((state, c), 0)
        if state == len(substring):
            occurrences.append(i - len(substring) + 1)
    
    return occurrences

def kmp_search(text, pattern):
    failure = [0] * len(pattern)
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[j] != pattern[i]:
            j = failure[j-1]
        if pattern[j] == pattern[i]:
            j += 1
        failure[i] = j

    j = 0
    locations = []
    for i in range(len(text)):
        while j > 0 and pattern[j] != text[i]:
            j = failure[j-1]
        if pattern[j] == text[i]:
            j += 1
        if j == len(pattern):
            locations.append(i - len(pattern) + 1)
            j = failure[j-1]

    return locations

def bmh_search(text, pattern):
    m = len(pattern)
    n = len(text)
    if m > n:
        return []

    # Preprocessing
    skip = {}
    for i in range(m - 1):
        skip[pattern[i]] = m - i - 1

    # Searching
    i = 0
    indexes = []
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j == -1:
            indexes.append(i)
            i += 1
        else:
            if text[i + j] in skip:
                i += skip[text[i + j]]
            else:
                i += m

    return indexes

def main():
    with open('english.50MB', 'r') as f:
        text = f.read()

    words = text.split()
    random_words = random.sample(words, 100)
    algorithms = [brute_serach, substring_search, kmp_search, bmh_search]

    for alg in algorithms:
        start = time.time()
        for word in random_words:
            alg(text, word)
        end = time.time()
        print(alg.__name__, end - start)

if __name__ == "__main__":
    main()

# 100 vyhledavani v 50MB textu non boosting x86 cpu
# brute_serach 1211.5342829227448
# substring_search 1247.644897222519
# kmp_search 814.9461803436279
# bmh_search 530.0323231220245

# 100 vyhledavani v 50MB textu arm64 cpu
# brute_serach 489.08472180366516
# substring_search 613.1546101570129
# kmp_search 344.8560268878937
# bmh_search 202.68151926994324
