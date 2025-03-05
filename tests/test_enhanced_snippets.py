from search_engine import search_knowledge_base, parse_boolean_query, extract_search_terms

def test_enhanced_snippets():
    """
    Test the enhanced snippet generation functionality of the search engine.
    This demonstrates how the search engine now shows multiple snippets when
    different search terms match in different parts of a document.
    """
    print("\n===== TESTING ENHANCED SNIPPET GENERATION =====\n")
    
    # Test 1: Simple query with multiple terms that might appear in different parts of documents
    query1 = "swami AND teachings"
    print(f"\nTest 1: Simple query with multiple terms: '{query1}'")
    print("This should show snippets containing both 'swami' and 'teachings' terms")
    search_knowledge_base(query1)
    
    # Test 2: Complex nested query with terms likely to appear in different parts
    query2 = "(swami AND anandashram) AND (discourse OR teachings)"
    print(f"\nTest 2: Complex nested query: '{query2}'")
    print("This should show snippets containing matches for all terms in the query")
    search_knowledge_base(query2)
    
    # Test 3: Query with terms that are likely to be far apart in documents
    query3 = "temple AND swami"
    print(f"\nTest 3: Query with potentially distant terms: '{query3}'")
    print("This should show separate snippets when 'temple' and 'swami' are far apart")
    search_knowledge_base(query3)
    
    # Demonstrate how search terms are extracted from a complex query
    complex_query = "((swami AND anandashram) OR (swami AND parijnanashram)) AND (teachings OR discourse)"
    parsed = parse_boolean_query(complex_query)
    terms = extract_search_terms(parsed)
    
    print("\n===== SEARCH TERM EXTRACTION DEMONSTRATION =====")
    print(f"Query: {complex_query}")
    print(f"Extracted search terms: {terms}")
    print("These terms are used to generate relevant snippets in the search results")

if __name__ == "__main__":
    test_enhanced_snippets() 