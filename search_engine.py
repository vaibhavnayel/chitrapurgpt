import re
from retrievers import load_docs_from_jsonl

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
        result = None
        current_operator = None
        i = start_idx
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '(':
                # Parse a nested expression
                sub_expr, next_idx = parse_expression(tokens, i + 1)
                
                if result is None:
                    result = sub_expr
                elif current_operator:
                    result = {current_operator: [result, sub_expr]}
                    current_operator = None
                
                i = next_idx
                continue
            
            elif token == ')':
                # End of the current expression
                return result, i + 1
            
            elif token in ('AND', 'OR'):
                current_operator = token
            
            else:
                # Regular search term
                if result is None:
                    result = token
                elif current_operator:
                    result = {current_operator: [result, token]}
                    current_operator = None
            
            i += 1
        
        return result, i
    
    parsed_query, _ = parse_expression(tokens)
    return parsed_query

def evaluate_query(query_expr, doc):
    """
    Evaluate a parsed boolean query against a document.
    Returns True if the document matches the query, False otherwise.
    """
    if isinstance(query_expr, str):
        # Simple term matching
        return query_expr.lower() in doc.page_content.lower().split()
    
    if isinstance(query_expr, dict):
        # Compound expression with AND/OR
        operator = list(query_expr.keys())[0]
        operands = query_expr[operator]
        
        if operator == 'AND':
            return evaluate_query(operands[0], doc) and evaluate_query(operands[1], doc)
        elif operator == 'OR':
            return evaluate_query(operands[0], doc) or evaluate_query(operands[1], doc)
    
    return False

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
    
    print(f"Found {len(matching_docs)} matching documents")
    
    # Extract all search terms from the query
    all_terms = extract_search_terms(parsed_query)
    
    # Print the titles and a snippet containing the matched terms
    result_string = ""
    for i, doc in enumerate(matching_docs, 1):
        title = doc.metadata.get('title', 'Untitled')
        source = doc.metadata.get('source', 'Unknown')
        
        # Generate a snippet containing the matched terms
        content_snippet = generate_snippet_with_matches(doc.page_content, all_terms)
        
        result_string += f"\n{i}. {title} (Source: {source})"
        result_string += f"\nSnippet: {content_snippet}"
        result_string += "\n\n"
    
    return result_string

def extract_search_terms(parsed_query):
    """
    Extract all search terms from a parsed query.
    """
    terms = []
    
    if isinstance(parsed_query, str):
        # Simple term
        terms.append(parsed_query.lower())
    elif isinstance(parsed_query, dict):
        # Compound expression
        operator = list(parsed_query.keys())[0]
        operands = parsed_query[operator]
        
        # Recursively extract terms from operands
        for operand in operands:
            terms.extend(extract_search_terms(operand))
    
    return terms

def generate_snippet_with_matches(content, search_terms, context_size=50, max_snippets=3):
    """
    Generate snippets from the content that contain the matched search terms.
    Includes context_size characters before and after each match.
    Combines up to max_snippets different matches to provide a comprehensive view.
    """
    content_lower = content.lower()
    snippets = []
    used_positions = set()  # Track positions we've already included in snippets
    
    for term in search_terms:
        term_lower = term.lower()
        start_pos = 0
        
        # Find all occurrences of this term
        while True:
            start_pos = content_lower.find(term_lower, start_pos)
            if start_pos == -1:
                break
                
            # Check if this position overlaps with any existing snippet
            overlap = False
            for used_start, used_end in used_positions:
                if (start_pos >= used_start and start_pos <= used_end) or \
                   (start_pos + len(term) >= used_start and start_pos + len(term) <= used_end):
                    overlap = True
                    break
            
            if not overlap:
                # Calculate the snippet boundaries with context
                snippet_start = max(0, start_pos - context_size)
                snippet_end = min(len(content), start_pos + len(term) + context_size)
                
                # Extract the snippet
                if snippet_start > 0:
                    prefix = "..."
                else:
                    prefix = ""
                    
                if snippet_end < len(content):
                    suffix = "..."
                else:
                    suffix = ""
                    
                snippet = prefix + content[snippet_start:snippet_end] + suffix
                snippets.append(snippet)
                
                # Mark this region as used - add as a tuple
                used_positions.add((snippet_start, snippet_end))
                
                # If we have enough snippets, stop
                if len(snippets) >= max_snippets:
                    break
            
            # Move past this occurrence
            start_pos += len(term)
    
    # If no snippets were found (which shouldn't happen since the document matched),
    # fall back to the first 200 characters
    if not snippets:
        return content[:200] + "..." if len(content) > 200 else content
    
    # Combine the snippets
    if len(snippets) == 1:
        return snippets[0]
    else:
        return " | ".join(snippets)

if __name__ == "__main__":
    query = "(viswesvara OR vishwehswara OR visweswara) AND (temple OR mandir)"
    print(f"Searching for: {query}")
    search_knowledge_base(query)