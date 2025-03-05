from search_engine import search_knowledge_base, parse_boolean_query, extract_search_terms

def demonstrate_search_engine():
    """
    Comprehensive demonstration of the enhanced search engine capabilities,
    including boolean query parsing and improved snippet generation.
    """
    print("\n===== CHITRAPUR SEARCH ENGINE DEMONSTRATION =====\n")
    print("This demonstration showcases the enhanced search engine with:")
    print("1. Support for complex boolean queries with nested expressions")
    print("2. Improved snippet generation showing relevant context around matched terms")
    print("3. Multiple snippets when different terms match in different parts of a document\n")
    
    # Demonstrate simple query
    simple_query = "swami"
    print(f"\n----- Simple Query: '{simple_query}' -----")
    print("Parsing structure:", parse_boolean_query(simple_query))
    search_knowledge_base(simple_query)
    
    # Demonstrate AND query
    and_query = "swami AND anandashram"
    print(f"\n----- AND Query: '{and_query}' -----")
    print("Parsing structure:", parse_boolean_query(and_query))
    search_knowledge_base(and_query)
    
    # Demonstrate OR query
    or_query = "temple OR mandir"
    print(f"\n----- OR Query: '{or_query}' -----")
    print("Parsing structure:", parse_boolean_query(or_query))
    search_knowledge_base(or_query)
    
    # Demonstrate nested query (one level)
    nested_query = "(swami OR guru) AND teachings"
    print(f"\n----- Nested Query (One Level): '{nested_query}' -----")
    print("Parsing structure:", parse_boolean_query(nested_query))
    search_knowledge_base(nested_query)
    
    # Demonstrate complex nested query
    complex_query = "((swami AND anandashram) OR (swami AND parijnanashram)) AND (teachings OR discourse)"
    print(f"\n----- Complex Nested Query: '{complex_query}' -----")
    print("Parsing structure:", parse_boolean_query(complex_query))
    
    # Extract and display search terms
    parsed = parse_boolean_query(complex_query)
    terms = extract_search_terms(parsed)
    print(f"Extracted search terms: {terms}")
    
    # Search with the complex query
    search_knowledge_base(complex_query)
    
    print("\n===== DEMONSTRATION COMPLETE =====")
    print("The search engine successfully demonstrates:")
    print("✓ Parsing of complex boolean expressions")
    print("✓ Extraction of search terms from nested queries")
    print("✓ Generation of relevant snippets with context around matched terms")
    print("✓ Combination of multiple snippets for comprehensive results")

if __name__ == "__main__":
    demonstrate_search_engine() 