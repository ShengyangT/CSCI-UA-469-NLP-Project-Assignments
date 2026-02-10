import argparse
import math
import re

from stop_list import closed_class_stop_words

stop_words = set(closed_class_stop_words)

STEM_RULES = [
    ("ization", "ize"),
    ("ational", "ate"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("iveness", "ive"),
    ("tional", "tion"),
    ("biliti", "ble"),
    ("logies", "logy"),
    ("ies", "y"),
    ("ied", "y"),
    ("ments", "ment"),
    ("ment", ""),
    ("ness", ""),
    ("ings", "ing"),
    ("ing", ""),
    ("edly", "e"),
    ("ed", ""),
    ("ally", "al"),
    ("ly", ""),
    ("ers", "er"),
    ("er", ""),
    ("ors", "or"),
    ("or", ""),
    ("als", "al"),
    ("al", ""),
    ("s", "")
]


def simple_stem(token):
    for suffix, replacement in STEM_RULES:
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[:-len(suffix)] + replacement
    return token

def load_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# Function to separate documents by their identifier from abstracts or queries
def parse_documents_by_id(text_content):
    return text_content.split('.I ')[1:]

# Function to extract different sections from a document
def extract_document_sections(document_lines):
    sections = {"title": "", "author": "", "bib_info": "", "abstract": ""}
    current_section = None

    for line in document_lines:
        line = line.strip()
        if line.startswith(".T"):
            current_section = "title"
        elif line.startswith(".A"):
            current_section = "author"
        elif line.startswith(".B"):
            current_section = "bib_info"
        elif line.startswith(".W"):
            current_section = "abstract"
        else:
            if current_section:
                sections[current_section] += line + " "

    return sections

# Function to process each document (abstracts) 
def parse_document_abstract(document_text):
    lines = document_text.strip().split("\n")
    document_id = lines[0]
    sections = extract_document_sections(lines[1:])
    title = sections.get("title", "").strip()
    abstract = sections.get("abstract", "").strip()
    combined_content = " ".join(segment for segment in [title, abstract] if segment)

    return {
        "doc_id": document_id,
        "title": title,
        "author": sections.get("author", "").strip(),
        "bib_info": sections.get("bib_info", "").strip(),
        "abstract": abstract,
        "content": combined_content.strip()
    }

# Process the abstracts from 'cran.all.1400'
def load_document_abstracts(filepath):
    content = load_file(filepath)
    documents = parse_documents_by_id(content)
    return [parse_document_abstract(doc) for doc in documents]

# Extract the actual query text
def get_query_content(query_lines):
    content = ""
    current_section = None

    for line in query_lines:
        line = line.strip()
        if line.startswith(".W"):
            current_section = "abstract"
        elif current_section == "abstract":
            content += line + " "

    return content

# Process each query
def parse_single_query(query_text, assigned_id):
    lines = query_text.strip().split("\n")
    query_id = lines[0].strip()
    content = get_query_content(lines[1:])
    return {
        "mapped_id": assigned_id,
        "query_id": query_id,
        "abstract": content.strip(),
        "content": content.strip()
    }

# Process the queries from 'cran.qry'
def load_cran_queries(filepath):
    content = load_file(filepath)
    queries = parse_documents_by_id(content)
    return [parse_single_query(query, idx + 1) for idx, query in enumerate(queries)]

# Preprocess text (tokenize, remove stop words and punctuation)
def clean_and_tokenize(text):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    cleaned_tokens = []

    for token in tokens:
        if token in stop_words:
            continue

        stemmed = simple_stem(token)
        if stemmed and stemmed not in stop_words:
            cleaned_tokens.append(stemmed)

    return cleaned_tokens

# Update document frequencies for each term
def increment_doc_frequencies(processed_document, doc_freq_dict):
    for term in set(processed_document):
        if term not in doc_freq_dict:
            doc_freq_dict[term] = 0
        doc_freq_dict[term] += 1

# Calculate document frequencies for all abstracts
def compute_doc_frequencies(document_list):
    doc_freq_dict = {}
    for doc in document_list:
        increment_doc_frequencies(clean_and_tokenize(doc.get('content', doc['abstract'])), doc_freq_dict)
    return doc_freq_dict

# Calculate term frequencies for an abstract
def compute_term_frequencies(processed_text):
    term_freq_dict = {}
    for word in processed_text:
        term_freq_dict[word] = term_freq_dict.get(word, 0) + 1
    return term_freq_dict

# Calculate inverse document frequency (IDF)
def compute_idf(df_count, total_document_count):
    return math.log((total_document_count + 1) / (df_count + 1)) + 1

# Calculate TF-IDF feature vector
def compute_feature_vector(term_freq_dict, doc_freq_dict, total_document_count):
    feature_vector = {}
    for word, raw_tf in term_freq_dict.items():
        if raw_tf <= 0:
            continue

        tf_weight = 1 + math.log(raw_tf)
        idf_value = compute_idf(doc_freq_dict.get(word, 0), total_document_count)
        if idf_value <= 0:
            continue

        feature_vector[word] = tf_weight * idf_value
    return feature_vector

# Calculate the TF-IDF vector for a given abstract/query
def compute_tf_idf_vector(input_text, doc_freq_dict, total_document_count):
    processed_text = clean_and_tokenize(input_text.get('content', input_text['abstract']))
    tf_dict = compute_term_frequencies(processed_text)
    return compute_feature_vector(tf_dict, doc_freq_dict, total_document_count)

# Calculate dot product for cosine similarity
def calculate_dot_product(query_vector, document_vector):
    return sum(query_vector[word] * document_vector.get(word, 0) for word in query_vector.keys())

# Calculate magnitude for cosine similarity
def calculate_magnitude(vector):
    return math.sqrt(sum(value ** 2 for value in vector.values()))

# Calculate cosine similarity between query and document vectors
def compute_cosine_similarity(query_vector, document_vector):
    dot_prod = calculate_dot_product(query_vector['TF-IDF'], document_vector['TF-IDF'])
    query_mag = calculate_magnitude(query_vector['TF-IDF'])
    doc_mag = calculate_magnitude(document_vector['TF-IDF'])

    if query_mag == 0 or doc_mag == 0:
        return 0
    return dot_prod / (query_mag * doc_mag)

# Argument parsing for file paths
def setup_argument_parser():
    parser = argparse.ArgumentParser(description="TF-IDF based Ad Hoc Information Retrieval System")
    parser.add_argument('--documents', type=str, required=True, help="Path to cran.all.1400 file (abstracts)")
    parser.add_argument('--queries', type=str, required=True, help="Path to cran.qry file (queries)")
    parser.add_argument('--output', type=str, required=True, help="Path to output file where results will be written")
    return parser.parse_args()

# Get top 100 results for a query
def retrieve_top_results(query_vector, document_list):
    all_similarities = [(doc['doc_id'], compute_cosine_similarity(query_vector, doc)) for doc in document_list]
    all_similarities.sort(key=lambda tup: tup[1], reverse=True)
    top_100 = all_similarities[:100]
    if len(top_100) < 100:
        top_100 += [(doc['doc_id'], 0) for doc in document_list[len(top_100):100]]
    return top_100

# Write the final results to the output file
def save_results_to_file(query_list, document_list, output_filepath):
    with open(output_filepath, 'w') as f:
        for query in query_list:
            results = retrieve_top_results(query, document_list)
            for doc_id, similarity_score in results:
                f.write(f"{query['mapped_id']} {doc_id} {similarity_score}\n")

# Main function to process documents and queries and calculate similarity
def main():
    args = setup_argument_parser()
    documents = load_document_abstracts(args.documents)  # Abstracts (cran.all.1400)
    queries = load_cran_queries(args.queries)  # Queries (cran.qry)
    doc_freq_dict = compute_doc_frequencies(documents)
    total_document_count = len(documents)

    for query in queries:
        query['TF-IDF'] = compute_tf_idf_vector(query, doc_freq_dict, total_document_count)

    for document in documents:
        document['TF-IDF'] = compute_tf_idf_vector(document, doc_freq_dict, total_document_count)

    save_results_to_file(queries, documents, args.output)

if __name__ == '__main__':
    main()
