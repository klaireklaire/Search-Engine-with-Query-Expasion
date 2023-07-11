#!/usr/bin/python3
import json
import time
import math
import sys
import getopt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import wordnet
import zlib
from base64 import b64encode, b64decode

stemmer = PorterStemmer()

def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def run_search(dict_file, postings_file, queries_file, results_file):
    # time search
    start_time = time.time()

    print("loading files from disk...")
    # dict file opened as f and loads the JSON data from the file
    with open(dict_file, 'r') as f:
        dictionary = json.loads(zlib.decompress(b64decode(json.load(f)['jsonzip'])))
    # opens the postings file in binary mode
    with open(postings_file,'r') as f:
        postings = json.loads(zlib.decompress(b64decode(json.load(f)['jsonzip'])))
        # postings = json.load(f)
    with open("length.txt", 'r') as f:
        doc_length = json.load(f)

    query = []
    relevant_docs = []

    print("processing query...")
    # tokenise the current query into sentences, then tokenise each sentence into words, flatten into a single list of words using a list of comprehension
    #Read the query on the file and tokenized into a queue
    with open(queries_file, 'r') as query_file:
        query_lines = query_file.readlines()
        query_content = query_lines[0]
        for phrase in sent_tokenize(query_content):
            query.append(phrase)
        # retrieve relevant documents
        feedback_content = query_lines[1:]
        for line in feedback_content:
            relevant_docs.append(sent_tokenize(line)[0])

    # TODO: Query Refinement
    # if query is only phrasal, return the documents answering to phrasal query
    # if there is free-text query, 
    # vectorize query 
    # query_vec = vectorize(query[0], dictionary, doc_length)
    # print(query_vec)
    # TODO: retrieve relevant documents 
    # vectorize relevant documents
    # docs_vec = []
    # obtain refined query
    # if len(relevant_docs) != 0:
    #     refined_query = rocchio_calculation(0.7, 0.3, query_vec, docs_vec)
    # continuing process free-text query to return result

    # check if all tokens are in the dictionary, if not, raise invalid query error
    allTokensValid = True
    for term in tokenise_query(query):
        # print(term)
        bypass = ['AND', '"', '“', "”", '\'\'', "``"]
        if term not in postings and term not in bypass:
            print("invalid query term: ", term)
            allTokensValid = False
            return

    print("retrieving relevant documents...")
    if allTokensValid:
        res = process_query(query, dictionary, postings, doc_length)

    # write results to file
    with open(results_file, 'w') as result_file:
        answer_list = res
        answer_list = list(map(int,answer_list))
        answer_list = sorted(answer_list)
        for answer in answer_list:
            result_file.write(str(answer) + " ")
        result_file.write("\n")

    # end time
    end_time = time.time()
    print("searching finished.")
    print(results_file, "generated.")
    print("search completed in", round(end_time - start_time, 5), "secs.")

def process_query(query, dictionary, postings, doc_length):
    """
    Return relevant docIDs for a given query

    Args:
        query (list): The input query in list form
        dictionary (dict): The list of words in the dictionary
        postings (dict): The list of postings
        doc_length (dict): The length of each document

    Returns:
        final_result (list): The final resulting list of docIDs
    """
    final_result = []
    free_text_res = []
    phrasal_res = []
    RPN_answer = []

    # retrieve query
    while query != []:
        first_out = query.pop(0)
        print(first_out)
        phrasal_query = []
        isPhrasal = False
        
        # check for boolean queries
        # count number of 'AND' operators
        AND_count = first_out.count('AND')
        # split up query tokens
        list_query = first_out.split(' AND ')
        for i in range(1, len(list_query) + AND_count, 2):
            list_query.insert(i, 'AND')

        # Phrasal Queries
        # if query contains double quotes, i.e. contains phrasal queries
        if '"' in first_out:
            isPhrasal = True
            # record start and end indices of the phrasal query
            indices = [ i for i, character in enumerate(first_out) if character == '"']
            # retrieve phrasal query using string slicing
            for index in range(0,len(indices),2):
                start = indices[index]+1
                end = indices[index+1]
                phrase = first_out[start:end]
                phrasal_query.append(phrase) # if len(phrasal_query) more than 1 -> probs have AND

        # if query contains phrasal queries, tokenize query correctly and return a new query
        if isPhrasal == True:
            new_query = []
            # [ " and the offer ", AND, vehicle] -> ["and the offer", AND, vehicl]
            for i in range(0, len(list_query)):
                if '"' not in list_query[i] and list_query[i] != 'AND':
                    tokens_list = word_tokenize(list_query[i])
                    for word in tokens_list:
                        new_query.append(word)
                elif list_query[i] == 'AND':
                    new_query.append(list_query[i])
                else:
                    phrase = phrasal_query.pop()
                    new_query.append(phrase)
        
        # if the query contains boolean query, run boolean query routine on it
        if 'AND' in first_out:
            if isPhrasal == True:
                # use new_query to shunting yard
                # has already been tokenised -> dont need to tokenise
                shunting_tokens = shunting_yard(new_query)
                RPN_answer = RPN(shunting_tokens, dictionary, postings, doc_length)

            else:
                # for free text queries: use Shunting Yard Algo + RPN to get the relevant docIDs
                list_query = tokenise_query(list_query)
                shunting_tokens = shunting_yard(list_query)
                RPN_answer = RPN(shunting_tokens, dictionary, postings, doc_length)
                
        # else, treat the query as either phrasal or free-text query
        else :
            # phrasal queries
            if isPhrasal == True:
                result_dict = {}
                # expand query using synonym
                phrasal_query = expand_query(new_query, dictionary)
                # find relevant docIDs for phrasal query
                phrasal_docid = find_docs_for_phrasal(phrasal_query, postings)
                # calculation of tf-idf for the query
                weights = calculate_tf_idf(phrasal_query, dictionary, doc_length)
                scores = calculate_scores(weights, dictionary, postings)
                for relevant_docs in phrasal_docid:
                    result_dict[relevant_docs] = scores[relevant_docs]
                # Sort the dictionary by value in descending order with key in ascending order as secondary condition
                sorted_result = sorted(result_dict.items(), key=lambda x: (-x[1], int(x[0])))
                phrasal_res = [x[0] for x in sorted_result]
                        
            # free text queries
            else:
                # expand query using synonym
                list_query = expand_query(list_query, dictionary)
                weights = calculate_tf_idf(list_query, dictionary, doc_length)
                scores = calculate_scores(weights, dictionary, postings)
                # Sort the dictionary by value in descending order with key in ascending order as secondary condition
                sorted_scores = sorted(scores.items(), key=lambda x: (-x[1], int(x[0])))
                free_text_res = [x[0] for x in sorted_scores]

    final_result = RPN_answer + phrasal_res + free_text_res
    return final_result

####################
# Query Refinement #
####################

# get q_m, the vectorized modified query
def rocchio_calculation(alpha, beta, query_vec, doc_vecs):
    weighted_query = alpha * query_vec
    weighted_centroid = beta * np.mean(doc_vecs, axis=0)
    refined_query_vec = weighted_query + weighted_centroid

    return refined_query_vec

def expand_query(query_tokens, dictionary):
    """
    Expands a query using WordNet ontology.

    Args:
        query (str): The input query to be expanded.
        dictionary (list): The list of words in the dictionary or thesaurus.

    Returns:
        str: The expanded query with synonyms from the ontology.
    """
    print("expanding query using synonyms...")
    res = []
    query_tokens = query_tokens[0].split(" ")
    # Expand query using WordNet
    expanded_query_tokens = []
    for token in query_tokens:
        # If token is in dictionary, add it to expanded query as is
        expanded_query_tokens.append(token)
        # Get synonyms from WordNet
        synonyms = []
        added = False # limit the number of synonyms
        for synset in wordnet.synsets(token):
            if added:
                break
            synonyms += synset.lemma_names()
            added = True

        # Add expanded synonyms to expanded query
        print(synonyms)
        expanded_query_tokens += synonyms

    # Remove duplicates and return expanded query
    expanded_query_tokens = list(set(expanded_query_tokens))
    expanded_query_text = " ".join(expanded_query_tokens)
    processed_query_tokens = set(tokenise_query([expanded_query_text]))
    # Remove words that are not in the dictionary
    for token in processed_query_tokens:
        if token in dictionary:
            res.append(token)

    print(res)

    return res

###################
# Phrasal Queries #
###################

def find_docs_for_phrasal(term_list, postings_file):
    dictionary = {}
    # print("term_list: " + term_list)
    term_list = tokenise_query([term_list])

    # get all the doc_id of the word from the postings
    for word in term_list:
        doc_id = []
        for docid in postings_file[word]:
            if word in dictionary.keys():
                doc_id.append(docid)
                dictionary[word] = doc_id

            else:
                doc_id.append(docid)
                dictionary[word] = doc_id

    # get the doc_ids where all the terms are present
    intersection_docids = set.intersection(*(set(val) for val in dictionary.values()))
    result = []

    for docid in intersection_docids:
        correct_phrase = [False] * (len(term_list) - 1)
        first_word = term_list[0]
        first_word_pos = postings_file[first_word][docid][2]
        # iterate the positions array
        for pos in first_word_pos:
            #compare it with the other terms in the phrasal query
            for i in range(1,len(term_list)):
                other_terms = term_list[i]
                other_terms_pos = postings_file[other_terms][docid][2]
                # indexing is based on word -> if first word position + i == second word position -> then correct phrase -> True
                if (pos + i) in other_terms_pos:
                    correct_phrase[i-1] = True
        if all(correct_phrase) == True:
            result.append(docid)

    return result

##########
# tf-idf #
##########

def calculate_scores(tfidf, dictionary, postings):
    """
    Calculate the scores of all documents in the collection based on the tf-idf weights of the query terms

    Arguments:
    tfidf: A dictionary object containing the tf-idf weights for each term in the query
    dictionary: A dictionary object containing the terms and their corresponding document frequencies and positions in the postings file
    postings_f: A file object for the postings file containing the lists of documents and their term frequencies

    Returns:
    A dictionary object containing the scores of all documents in the collection
    """
    scores = {}

    for term in tfidf.keys():
        postings_list = postings[term]
        for docID, posting in postings_list.items(): # posting = {'246427': [0.02358525529099843, 'content', [3143]]}
            freq = posting[0]
            if docID not in scores: 
                scores[docID] = freq*tfidf[term]
            else:
                scores[docID] += freq*tfidf[term]

    return scores

def calculate_tf_idf(term_list, dictionary, doc_length):
    """
    Parse the input query and calculate the tf-idf weights for each term in the query

    Arguments:
    term_list: A list of terms
    dictionary: A dictionary object containing the terms and their corresponding document frequencies and positions in the postings file
    doc_length: A dictionary object containing the lengths of each document in the collection

    Returns:
    A dictionary object containing the tf-idf weights for each term in the query
    """
    tfidf = vectorize(term_list, dictionary, doc_length)

    # cosine length normalisation
    cos_length = sum([tfidf**2 for tfidf in tfidf.values()])**0.5
    for term in tfidf.keys():
        if tfidf[term] != 0:
            tfidf[term] = tfidf[term]/cos_length

    return tfidf

def vectorize(text, dictionary, doc_length):
    # if the text is a list already tokenized, skip tokenizing
    if isinstance(text, list):
        sen_list = text
    else:
        sen_list = sent_tokenize(text)
    word_list = []
    for sentence in sen_list:
        words = word_tokenize(sentence)
        for word in words:
            word_list.append(word)
    term_list = ([stemmer.stem(word) for word in word_list])

    # counting raw tf
    tf = {}
    for term in term_list:
        if term not in tf:
            tf[term] = 1
        else:
            tf[term] += 1

    # converting raw tf to log tf
    for term in tf.keys():
        tf[term] = 1 + math.log(tf[term], 10)

    # tfidf
    tfidf = {}
    for term in tf.keys():
        if term not in dictionary:
            continue
        else:
            df = dictionary[term][0]
            idf = math.log(len(doc_length)/df, 10)
            tfidf[term] = tf[term]*idf

    return tfidf

def tokenise_query(query):
    # parsing a query
    sen_list = query
    word_list = []
    for sentence in sen_list:
        words = word_tokenize(sentence)
        for word in words:
            word_list.append(word)

    term_list = []
    for word in word_list:
        if word == 'AND':
            term_list.append(word)
        else:
            term_list.append(stemmer.stem(word))

    return term_list

###################
# Boolean Queries #
###################

# # returns the function of and when there are two lists list1: [1 3 5], list2: [3 6 9] -> return [3]
def operator_function_AND(first_list, second_list):
    #first_list is in terms of integers -> check
    first_list = list(map(int,first_list))
    second_list = list(map(int,second_list))
    answer_list = list(set(first_list).intersection(second_list))
    return answer_list

def shunting_yard(queries):
    stack = []
    output = []
    list_query = queries

    while list_query != []:
        first = list_query.pop(0) # pop from the start
        # if it's operand -> push it into output
        if is_operand(first):
            output.append(first)
        # if it's operator -> push it into operator stack 
        elif is_operator(first):
            stack.append(first)

    while stack != []:
        operator = stack.pop()
        output.append(operator)

    return output

def RPN(shunted_queries, dictionary, postings, doc_length):
    intermediate_res = []

    while shunted_queries != []:
        word = shunted_queries.pop(0)
        # operand -> put inside intermediate_res until it operator comes
        if is_operand(word) and " " not in word:
            # get the doc_id of the word
            docid_list = []
            for key, value in postings[word].items():
                docid_list.append(key)
            intermediate_res.append(docid_list)

        elif is_operator(word):
            first_list = intermediate_res.pop() # top stack
            second_list = intermediate_res.pop() # second top stack
            # AND Operator
            # Order doesnt matter in the AND operator
            result_list = operator_function_AND(first_list, second_list)
            intermediate_res.append(result_list)

        # Phrasal queries -> use different search
        # have a space -> phrasal queries
        elif is_operand(word) and " " in word:
            doc_id = find_docs_for_phrasal(word,postings)
            intermediate_res.append(doc_id)
    
    # result is the last remaining in the stack
    if len(intermediate_res) != 0:
        result = intermediate_res[0]
    else: result = []

    return result

# does not contain all the 5 operators
def is_operand(word):
    if word != "AND":
        return True
    return False
    
#"(" and ")" are not considered operator -> for shunting yard
def is_operator(word):
    if word == "AND":
        return True
    return False

dict_file = 'dictionary.txt'
postings_file = 'postings.txt'
queries_file = 'queries.txt'
results_file = 'results.txt'

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dict_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        queries_file = a
    elif o == '-o':
        results_file = a
    else:
        assert False, "unhandled option"

if dict_file == None or postings_file == None or queries_file == None or results_file == None :
    usage()
    sys.exit(2)
   

run_search(dict_file, postings_file, queries_file, results_file)
