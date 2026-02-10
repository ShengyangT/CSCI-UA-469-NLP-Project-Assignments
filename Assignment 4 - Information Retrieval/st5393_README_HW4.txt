# TF-IDF Information Retrieval System

## System Overview

This system implements a TF-IDF (Term Frequency-Inverse Document Frequency) based ad-hoc information retrieval system that ranks documents according to their relevance to user queries. The system uses cosine similarity to calculate the similarity between queries and documents, returning the most relevant documents.

## Technical Methods

### 1. Text Processing
- **Tokenization**: Uses a lightweight regular-expression tokenizer to keep only alphabetic word spans
- **Lowercasing**: Converts all text to lowercase prior to any other processing
- **Stop Word Filtering**: Drops every token present in the supplied stop list
- **Custom Stemming**: Applies a compact suffix-based stemmer tailored for the Cranfield vocabulary (no external libraries required)
- **Digit Filtering**: Removes any purely numeric token

### 2. Document Representation Enhancements
- **Title + Abstract Fusion**: Concatenates each document title with its abstract so high-signal title terms participate in scoring
- **Query Normalization**: Applies the same preprocessing pipeline to queries and documents to maintain comparable vector spaces

### 3. TF-IDF Calculation
- **Log-scaled Term Frequency**: Weights each term as `1 + log(tf)` to temper long-document bias while retaining frequency cues
- **Smoothed Inverse Document Frequency**: Computes `log((N + 1) / (df + 1)) + 1` to avoid zero weights and emphasize distinctive terms
- **Sparse Vector Construction**: Stores only non-zero TF-IDF weights per query/document for efficient cosine evaluation

### 4. Similarity Calculation
- **Vectorization**: Converts queries and documents into TF-IDF vectors
- **Cosine Similarity**: Calculates cosine similarity between query and document vectors
- **Ranking**: Sorts documents by similarity score in descending order

## System Requirements

- Python 3.x (tested with Python 3.9)
- No third-party packages are required beyond the standard library


## Usage Instructions

### Basic Command
```bash
python st5393_main_HW4.py --documents cran.all.1400 --queries cran.qry --output output.txt
```

### Parameter Description
- `--documents`: Specify the path to the document file (required)
- `--queries`: Specify the path to the query file (required)
- `--output`: Specify the path to the output file (required)

## Output Format

Each line in the output file follows the format:
```
<Query_ID> <Document_ID> <Similarity_Score>
```

### Output Example
```
1 13 0.25000919314485215
1 486 0.17187396039846603
1 51 0.16332441793924984
2 764 0.17626161075730074
2 239 0.11880402044822909
```

### Output Description
- **Query ID**: 1-225, representing the query number
- **Document ID**: 1-1400, representing the document number
- **Similarity Score**: Floating-point number between 0-1, higher scores indicate stronger relevance
- **Result Count**: Top 100 most relevant documents returned for each query
- **Total Lines**: 22,500 lines (225 queries × 100 results)

## Algorithm Workflow

1. **Document Parsing**: Extract title, author, abstract, and other information from cran.all.1400 file, then merge the title and abstract so title terms contribute to scoring
2. **Query Parsing**: Extract query text from cran.qry file
3. **Text Preprocessing**: Perform regex tokenization, stop word removal, digit filtering, and light stemming on all text
4. **Document Frequency Calculation**: Count how many documents each stemmed word appears in
5. **TF-IDF Vectorization**: Calculate log-scaled TF weights and smoothed IDF values for each document and query
6. **Similarity Calculation**: Use cosine similarity to calculate relevance between queries and documents
7. **Result Ranking**: Sort by similarity score in descending order, take the top 100 results per query (padding with zero-score entries if fewer than 100 matches are found)
8. **Result Output**: Write results to the specified output file

## Performance Characteristics

- **Retrieval Precision**: Based on classic TF-IDF information retrieval methods
- **Processing Speed**: Suitable for medium-scale document collections
- **Scalability**: Supports document and query collections of different sizes
- **Standardization**: Uses standard TF-IDF and cosine similarity algorithms

## Important Notes

1. Ensure all input files exist and are in the correct format
2. Output files will be overwritten, please backup important results
3. Processing time depends on the number of documents and queries
4. A similarity score of 0 indicates complete irrelevance between query and document
