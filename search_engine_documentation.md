# Boolean Search Engine Documentation

## Overview

This document provides an overview of the boolean search engine implementation in `search_engine.py`. The search engine supports boolean queries with nested parentheses, AND, and OR operations against a knowledge base of documents.

## Key Components

### 1. Query Tokenization

The `tokenize_query` function breaks a query string into tokens:
- Search terms (e.g., "temple", "swami")
- Boolean operators ("AND", "OR")
- Parentheses for grouping expressions

```python
def tokenize_query(query: str):
    """
    Tokenize a boolean query into tokens (terms, operators, and parentheses)
    """
    # Replace parentheses with spaces around them to make them separate tokens
    query = re.sub(r'\(', ' ( ', query)
    query = re.sub(r'\)', ' ) ', query)
    
    # Split by whitespace and filter out empty strings
    tokens = [token for token in query.split() if token]
    return tokens
```

### 2. Query Parsing

The `parse_boolean_query` function converts the tokenized query into a structured representation:
- Simple terms remain as strings
- Compound expressions are represented as dictionaries with the operator as the key and operands as a list

```python
def parse_boolean_query(query: str):
    """
    Parse a boolean query with support for nested parentheses, AND, and OR operators.
    Returns a structured representation of the query.
    """
    tokens = tokenize_query(query)
    
    def parse_expression(tokens, start_idx=0):
        """
        Recursively parse an expression from tokens starting at start_idx
        Returns (parsed_expression, next_idx)
        """
        # Recursive parsing implementation
        # ...
    
    parsed_query, _ = parse_expression(tokens)
    return parsed_query
```

### 3. Query Evaluation

The `evaluate_query` function evaluates a parsed query against a document:
- For simple terms, it checks if the term is in the document content
- For compound expressions, it recursively evaluates the operands and applies the boolean operator

```python
def evaluate_query(query_expr, doc):
    """
    Evaluate a parsed boolean query against a document.
    Returns True if the document matches the query, False otherwise.
    """
    if isinstance(query_expr, str):
        # Simple term matching
        return query_expr.lower() in doc.page_content.lower()
    
    if isinstance(query_expr, dict):
        # Compound expression with AND/OR
        operator = list(query_expr.keys())[0]
        operands = query_expr[operator]
        
        if operator == 'AND':
            return evaluate_query(operands[0], doc) and evaluate_query(operands[1], doc)
        elif operator == 'OR':
            return evaluate_query(operands[0], doc) or evaluate_query(operands[1], doc)
    
    return False
```

### 4. Knowledge Base Search

The `search_knowledge_base` function ties everything together:
- Loads documents from a JSONL file
- Parses the query
- Evaluates the query against each document
- Returns and displays matching documents

```python
def search_knowledge_base(query: str):
    """
    Search the knowledge base for documents matching the boolean query.
    Supports nested parentheses, AND, and OR operators.
    """
    docs = load_docs_from_jsonl("knowledge_base.jsonl")
    parsed_query = parse_boolean_query(query)
    
    matching_docs = []
    for doc in docs:
        if evaluate_query(parsed_query, doc):
            matching_docs.append(doc)
    
    # Display results
    # ...
    
    return matching_docs
```

## Enhanced Snippet Generation

The search engine has been enhanced to provide more relevant and comprehensive snippets in search results. Instead of simply displaying the first 200 characters of a matching document, the engine now:

1. **Extracts all search terms** from the parsed query structure using a recursive function that handles both simple terms and complex nested expressions.

2. **Generates contextual snippets** that contain the matched search terms, providing context around each match (50 characters before and after by default).

3. **Combines multiple snippets** when different search terms match in different parts of a document, providing a more comprehensive view of the document's relevance to the query.

4. **Avoids overlapping snippets** by tracking which regions of the document have already been included in the result.

5. **Limits the number of snippets** to prevent excessively long results (default is 3 snippets per document).

### Implementation Details

The enhanced snippet generation is implemented through two key functions:

1. `extract_search_terms(parsed_query)`: Recursively extracts all search terms from a parsed query structure, handling both simple terms and complex nested expressions.

2. `generate_snippet_with_matches(content, search_terms, context_size=50, max_snippets=3)`: Generates snippets from the document content that contain the matched search terms, with context around each match, and combines up to a specified maximum number of snippets.

### Example

For a complex query like:
```
((swami AND anandashram) OR (swami AND parijnanashram)) AND (teachings OR discourse)
```

The search engine:
1. Extracts all search terms: `['swami', 'anandashram', 'swami', 'parijnanashram', 'teachings', 'discourse']`
2. Finds occurrences of these terms in each document
3. Generates snippets with context around each match
4. Combines multiple snippets when different terms match in different parts of the document
5. Separates multiple snippets with a " | " delimiter for readability

This enhancement significantly improves the relevance of search results by showing users exactly where and how their search terms match in the documents, rather than just showing the beginning of the document.

## Query Examples

### Simple Queries

- `temple` - Searches for documents containing "temple"
- `temple AND mandir` - Searches for documents containing both "temple" and "mandir"
- `temple OR mandir` - Searches for documents containing either "temple" or "mandir"

### Nested Queries (One Level)

- `(temple OR mandir) AND swami` - Searches for documents containing either "temple" or "mandir", and also containing "swami"
- `temple AND (swami OR guru)` - Searches for documents containing "temple" and also containing either "swami" or "guru"

### Nested Queries (Multiple Levels)

- `((temple OR mandir) AND (swami OR guru))` - Searches for documents containing either "temple" or "mandir", and also containing either "swami" or "guru"
- `(vishweshwara AND (temple OR (mandir AND shrine)))` - Searches for documents containing "vishweshwara" and also containing either "temple" or both "mandir" and "shrine"
- `((swami AND anandashram) OR (swami AND parijnanashram))` - Searches for documents containing either both "swami" and "anandashram", or both "swami" and "parijnanashram"

## Parsed Query Structure

The parser converts queries into a structured representation:

1. Simple term: `"temple"` → `"temple"`
2. Simple AND: `"temple AND mandir"` → `{"AND": ["temple", "mandir"]}`
3. Simple OR: `"temple OR mandir"` → `{"OR": ["temple", "mandir"]}`
4. Nested query: `"(temple OR mandir) AND swami"` → `{"AND": [{"OR": ["temple", "mandir"]}, "swami"]}`
5. Complex nested query: `"((swami AND anandashram) OR (swami AND parijnanashram))"` → `{"OR": [{"AND": ["swami", "anandashram"]}, {"AND": ["swami", "parijnanashram"]}]}`

## Limitations and Future Improvements

1. The current implementation only supports binary operations (each operator has exactly two operands).
2. The search is case-insensitive but does not support stemming or fuzzy matching.
3. The implementation could be extended to support more operators (NOT, XOR, etc.).
4. Performance could be improved by indexing the documents before searching.
5. The search could be enhanced with relevance scoring to rank results.

## Conclusion

The boolean search engine provides a powerful way to search a knowledge base using complex queries with nested expressions. It demonstrates the use of recursive parsing and evaluation to handle arbitrarily complex boolean expressions. 