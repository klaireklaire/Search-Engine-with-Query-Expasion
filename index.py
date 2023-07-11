#!/usr/bin/python3
import sys
import getopt
import os
import csv
import json
import math
import string
import zlib, base64
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
punc = set(string.punctuation)
PUNCT = "!@#$%^&*()-_{+=[];:'\"<>?,./0123456789"
COURT_HIERARCHY = {"MOST_IMPT" : ["SG Court of Appeal", "SG Privy Council", "UK House of Lords", "UK Supreme Court", 
                                  "High Court of Australia", "CA Supreme Court"],
                   "IMPT"      : ["SG High Court", "Singapore International Commercial Court", "HK High Court", 
                                  "HK Court of First Instance", "UK Crown Court", "UK Court of Appeal", "UK High Court",
                                  "Federal Court of Australia", "NSW Court of Appeal", "NSW Court of Criminal Appeal",
                                  "NSW Supreme Court"]}

# document variables
docID = court_weight = 0

JSONZIPKEY = 'jsonzip'

# define the zones and fields for each document
def parse_document(row):
    global docID, court_weight
    zones = {}
    zones['docID'] = docID = row[0]
    zones['title'] = row[1]
    zones['content'] = row[2]
    zones['date_posted'] = row[3]
    zones['court'] = row[4]

    # handle court, give weight to document depending on court hierarchy
    if COURT_HIERARCHY["MOST_IMPT"].count(row[4]) > 0:
        court_weight = 0.01
    elif COURT_HIERARCHY["IMPT"].count(row[4]) > 0:
        court_weight = 0.005
    return zones

def build_index(csv_file, out_dict, out_postings):
    """
    Build an index from documents stored in a CSV file,
    then output the dictionary file and postings file.
    """
    print('Indexing...')

    dictionary = {}
    postings_list = {}
    doc_length = {}

    with open(csv_file, newline='', encoding="utf8", errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        csv.field_size_limit(sys.maxsize)
        next(reader)  # skip header row
        for i, row in enumerate(reader):
            zones = parse_document(row)

            # process each zone/field separately
            for zone, content in zones.items():
                # only text in title and content goes into the dictionary
                if (zone == 'docID') or (zone == 'court') or (zone == 'date_posted'):
                    continue

                # build list of terms
                tf = {}
                positions = {}
                word_list = [word_tokenize(t) for t in sent_tokenize(trim(content))]
                flattened_list = [item for sublist in word_list for item in sublist]
                term_list = [stemmer.stem(word.casefold()) for word in flattened_list if word not in punc]
                
                for j, word in enumerate(flattened_list):
                    if word in punc:
                        continue    # ignore punctuation
                    
                    term = stemmer.stem(word.casefold())
                    # counting raw tf and positions
                    if term not in tf:
                        tf[term] = 1
                        positions[term] = [j]
                    else:
                        tf[term] += 1
                        positions[term].append(j)
                
                # natural length of doc
                length = len(term_list)
                if zone == 'content':
                    doc_length[docID] = length

                # converting raw tf to log tf and computing doc vector length
                sum = 0
                for term in tf:
                    tf[term] = 1 + math.log(tf[term], 10)
                    sum += tf[term]**2

                # cosine length normalisation and accounting for court hierarchy weight
                cos_length = sum**0.5
                for term in tf:
                    tf[term] = tf[term]/cos_length + court_weight

                # writing dictionary
                for term in tf.keys():
                    if term not in dictionary:
                        dictionary[term] = [1]
                    else:
                        dictionary[term][0] += 1

                # writing postings list
                for term, freq in tf.items():
                    posting = {docID: (freq, zone, positions[term])}
                    if term not in postings_list:
                        postings_list[term] = posting
                    else:
                        postings_list[term].update(posting)

    # sort the postings and dictionary
    sorted_postings = dict(sorted(postings_list.items(), key=lambda x: x[0].lower()))
    sorted_dict = dict(sorted(dictionary.items(), key=lambda x: x[0].lower()))
    sorted_length = dict(sorted(doc_length.items(), key=lambda x: x[0].lower()))

    # write files to disk
    with open(out_postings, 'a', encoding='utf-8') as f1:
        for term in sorted_postings:    # store position of term in posting list into dictionary
            pos = f1.tell()
            dictionary[term].append(pos)

        zipped_postings = { JSONZIPKEY : base64.b64encode(zlib.compress(
                json.dumps(sorted_postings).encode('utf-8')
            )
        ).decode('ascii')
        }

        json.dump(zipped_postings, f1)

    with open(out_dict, 'w', encoding='utf-8') as f2:
        zipped_dict = { JSONZIPKEY : base64.b64encode(zlib.compress(
                json.dumps(sorted_dict).encode('utf-8')
            )
        ).decode('ascii')
        }
        json.dump(zipped_dict, f2)

    with open('length.txt', 'w', encoding='utf-8') as f3:
        json.dump(sorted_length, f3)

    print('Finished indexing.')


def trim(string):
    """
    Helper method to remove punctuation from a string.

    Parameters
    ----------
    string: str
        The string, with punctuation and numbers to remove
    """

    table = string.maketrans("", "", PUNCT)
    str = string.translate(table).replace("  ", " ")
    return str

def usage():
    print("usage: " +
          sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
