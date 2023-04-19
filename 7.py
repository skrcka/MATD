import numpy as np
import re
import nltk
from nltk.stem import PorterStemmer
import os
import math
from collections import defaultdict
import pprint


def documents_to_list(path):
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


def compute_term_freq(path, file_indices):
    dir_list = os.listdir(path)
    frequencies_dict = {}
    for file in dir_list:
        index = file_indices[file]
        with open(path+os.sep+file, "r", encoding='utf8') as f:
            text = f.read()
        f.close()
        for word in text.split():
            if word not in frequencies_dict:
                frequencies_dict[word] = {}
            if index not in frequencies_dict[word]:
                frequencies_dict[word][index] = 0
            frequencies_dict[word][index] += 1

    return frequencies_dict


def compute_inv_doc_freq(file_indices, dict):
    inv_doc_frequencies = {}
    N = len(file_indices)
    for word in dict:
        df_t = len(dict[word])
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
    for word in query:
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

def query(words, dict):
    fileSets = []
    for word in words:
        if word in dict:
            fileSets.append(dict[word])
        else:
            fileSets.append(set()) 
    fileSets.sort(key=len, reverse=False)
    print("fileSets for query: "+str(words))
    print(fileSets)
    if words.__len__() == 1:
        return fileSets[0]
    else:
        return fileSets[0].intersection(*fileSets[1:])


def normalize_vector(vec):
    """Normalize a vector to unit length."""
    norm = math.sqrt(sum(x**2 for x in vec))
    return [x / norm for x in vec]


def dot_product(vec1, vec2):
    """Compute the dot product of two vectors."""
    return sum(x * y for x, y in zip(vec1, vec2))


def dot_product_dict(dict1, dict2):
    """Compute the dot product of two dictionaries."""
    result = 0
    for key in set(dict1) & set(dict2):
        result += dict1[key] * dict2[key]
    return result


def normalize_matrix_by_rows(matrix):
    """Normalize a matrix by rows to unit length."""
    result = {}
    for rowkey in matrix:
        row = matrix[rowkey]
        norm = math.sqrt(sum(x**2 for x in row.values()))
        if norm != 0:
            result[rowkey] = {colkey: row[colkey] / norm for colkey in row}
    return result


def normalize_matrix_by_cols(matrix):
    """Normalize a matrix by columns to unit length."""
    result = {}
    cols = list(matrix.values())[0].keys()
    for col in cols:
        values = [matrix[row][col] for row in matrix]
        norm = math.sqrt(sum(x**2 for x in values))
        result[col] = {row: matrix[row][col] / norm for row in matrix}
    return result


def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity of two vectors."""
    return dot_product(vec1, vec2) 
    #/ (math.sqrt(dot_product(vec1, vec1)) * math.sqrt(dot_product(vec2, vec2)))


def cosine_similarity_dict(vec1, vec2):
    """Compute the cosine similarity of two vectors."""
    return dot_product_dict(vec1, vec2) 
    #/ (math.sqrt(dot_product_dict(vec1, vec1)) * math.sqrt(dot_product_dict(vec2, vec2)))


def rotate_matrix(matrix):
    """Rotate a matrix by 90 degrees."""
    flipped = {}
    for key, val in matrix.items():
        for subkey, subval in val.items():
            if subkey not in flipped:
                flipped[subkey] = {}
            flipped[subkey][key] = subval
    return flipped


def find_most_similiar_dictionaries(dict):
    """Find the most similiar dictionaries in a matrix."""
    """Format: {key1: (key2, similarity))}"""
    result = {}
    for key1 in dict:
        for key2 in dict:
            if key1 != key2:
                similarity = cosine_similarity_dict(dict[key1], dict[key2])
                if key1 not in result:
                    result[key1] = (key2, similarity)
                elif similarity > result[key1][1]:
                    result[key1] = (key2, similarity)
        #print(str(key1)+" done")
    return result


if __name__ == "__main__":
    targetFolder = "gutenberg_stemmed"
    #targetFolder = "test2_stemmed"
    runTests = False
    runQueries = False
    runVectorSpace = True

    MATDpath = os.getcwd() + os.sep + 'MATD'
    path = MATDpath + os.sep + targetFolder

    if not os.path.exists(path):
        print(path + " does not exist. Returning...")
        quit()

    indices, dict = documents_to_list(path)
    print("documents_to_list done")
    term_freq = compute_term_freq(path, indices)
    print("compute_term_freq done")
    inv_doc_freq = compute_inv_doc_freq(indices, dict)
    print("compute_inv_doc_freq done")
    tfidf = compute_tfidf(term_freq, inv_doc_freq)
    print("compute_tfidf done")
    tf_r = rotate_matrix(term_freq)
    tfidf_r = rotate_matrix(tfidf)

    print("\nIndices:")
    print(indices)
    print("")

    if(len(dict) < 100):
        print("Term_Freq:")
        print(term_freq)
        print("")
        print("INV_DOC_Freq:")
        print(inv_doc_freq)
        print("")
        print("tfIdf:")
        print(tfidf)

        print("")
        #print(dict)
    if(runQueries):
        print("_________________________________________________")
        query_list = ["sprog", "fro"]
        scores = score(tfidf, indices,query_list)
        print(query(query_list, dict))
        printTopScores(scores, indices, 10)
        print("_________________________________________________")

        query_list = ["my", "bitch","cuckold"]
        scores = score(tfidf, indices,query_list)
        print(query(query_list, dict))
        printTopScores(scores, indices, 10)
        print("_________________________________________________")

        query_list = ["my", "cocksuck"]
        scores = score(tfidf, indices,query_list)
        print(query(query_list, dict))
        printTopScores(scores, indices, 10)
        print("_________________________________________________")

        query_list = ["project", "thi","down"]
        scores = score(tfidf, indices,query_list)
        print(query(query_list, dict))
        printTopScores(scores, indices, 10)
        print("_________________________________________________")

        query_list = ["god"]
        scores = score(tfidf, indices,query_list)
        print(query(query_list, dict))
        printTopScores(scores, indices, 10)
        print("_________________________________________________")
    if(runVectorSpace):
        """print("_________________________________________________")
        print("tfidf similiarities (words):")
        print(find_most_similiar_dictionaries(normalize_matrix_by_rows(tfidf)))
        print("_________________________________________________")
        print("tf similiarities (words):")
        print(find_most_similiar_dictionaries(normalize_matrix_by_rows(term_freq)))"""
        print("_________________________________________________")
        print("tfidf_r similiarities (documents):")
        print(find_most_similiar_dictionaries(normalize_matrix_by_rows(tfidf_r)))
        print("_________________________________________________")
        print("tf_r similiarities (documents):")
        print(find_most_similiar_dictionaries(normalize_matrix_by_rows(tf_r)))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")
        print("documentNames:")
        pprint.pprint(indices)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++")



    #TEST BLOCK
    if(runTests):
        vector1 = [0,6,8]
        vector2 = [6,7,1]

        vector11 = [0, 0, 0, 6, 8]
        vector12 = [0, 0, 6, 7, 1]
        print(vector1,"->",normalize_vector(vector1))
        print(vector2,"->",normalize_vector(vector2))

        print(dot_product(normalize_vector(vector1),normalize_vector(vector2)))
        print(dot_product(normalize_vector(vector1),normalize_vector(vector1)))

        print(cosine_similarity(vector1,vector2))

        print(vector11,"->",normalize_vector(vector11))
        print(vector12,"->",normalize_vector(vector12))

        print(dot_product(normalize_vector(vector11),normalize_vector(vector12)))
        print(dot_product(normalize_vector(vector11),normalize_vector(vector11)))

        print(cosine_similarity(vector11,vector12))


        print("\nDictionary Playground")
        print("_________________________________________________")
        print("tfidf:")
        print(tfidf)
        print(normalize_matrix_by_rows(tfidf))
        print("_________________________________________________")
        print("term_freq:")
        print(term_freq)
        print(normalize_matrix_by_rows(term_freq))    
        print("_________________________________________________")
        print("tfidf_rotated:")
        print(tfidf_r)
        print(normalize_matrix_by_rows(tfidf_r))    
        print("_________________________________________________")
        print("tf_rotated:")
        print(tf_r)
        print(normalize_matrix_by_rows(tf_r))   
        print("_________________________________________________")
        

        #print(np.log(3))
        #print(np.log(1.5))
        #print(np.log(1))

        #print(np.log2(3))
        #print(np.log2(1.5))
        #print(np.log2(1))

        #print(np.log10(3))
        #print(np.log10(1.5))
        #print(np.log10(1))

    #input("Press Enter to continue...")


# max docs = 100
# 50, 20 , 5
# prvni pripad: 50*20 = 70, vyplivne nejhure 20 -> 20*5 = 25
# druhy pripad: 20*5 = 25, vyplivne nejhure 5 -> 5*50 = 55

# zmenit to na O(l1+l2) misto O(l1*l2)