== General Notes about this program ==

The program consists of two main components: indexing and searching. I (Klaire Pham) am in charge of the searching functionalities.

Indexing:
- Preprocess the documents by stemming and tokenizing the text
- For each document, store and process each field separately. The dictionary is comprised of information in the title and content    
  fields only
- Calculate the document length, term frequency, and cosine normalization for each term in each document
- Additional weight is given to documents based on the court hierarchy provided
- Store the resulting dictionary and postings list using the JSON module

Searching:
The searching works by dividing the query into 4 different cases:
1. Case 1: Boolean Queries containing Phrasal
2. Case 2: Boolean Queries containing Free Text Only
3. Case 3: Non-boolean Queries containing Phrasal Only
4. Case 4: Non-boolean Queries containing Free Text Only
The function process_query will give the resulting document IDs which is relevant
to the query based on the tf-idf computation.

In order to determine whether a query is phrasal is due to the presence of ‘“‘ in them. 
Once phrasal query is detected, the function find_docs_for_phrasal will find the documents
for which all the terms of the phrasal query exists. If such documents exists, 
the function will also verify the validity of the phrasal query though the positioning 
of each term.

In order to determine whether a query is boolean is due to the presence of 'AND' in them.
Once boolean query is detected, the query will be split based on (' AND ') and the function 
shunting_yard and RPN (Reverse Polish Notation) will be sort the queries and obtain the document
IDs for which the both terms are present.

Searching included partially implemented Query Refinement and fully implemented Query Expansion. These techniques are explained further in the BONUS.docx file. 



== Files included ==

README.txt      : The README document about this submission.
index.py        : The file to index the dataset.
search.py       : The file to run searches on queries.
dictionary.txt  : The dictionary file.
postings.txt    : The file containing all the posting lists.
length.txt      : The file containing the lenths of all the documents.
