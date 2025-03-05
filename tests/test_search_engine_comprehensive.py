from search_engine import tokenize_query, parse_boolean_query, evaluate_query, search_knowledge_base

def test_parser():
    """Test the query parser with various types of queries"""
    print("\n=== TESTING QUERY PARSER ===")
    
    # Simple queries
    print("\nSimple Queries:")
    queries = [
        "temple",
        "temple AND mandir",
        "temple OR mandir",
        "swami AND guru"
    ]
    
    for query in queries:
        tokens = tokenize_query(query)
        parsed = parse_boolean_query(query)
        print(f"Query: '{query}'")
        print(f"Tokens: {tokens}")
        print(f"Parsed: {parsed}")
        print("-" * 40)
    
    # Nested queries - one level
    print("\nNested Queries (One Level):")
    queries = [
        "(temple OR mandir) AND swami",
        "temple AND (swami OR guru)",
        "(swami AND anandashram) OR (guru AND teachings)"
    ]
    
    for query in queries:
        tokens = tokenize_query(query)
        parsed = parse_boolean_query(query)
        print(f"Query: '{query}'")
        print(f"Tokens: {tokens}")
        print(f"Parsed: {parsed}")
        print("-" * 40)
    
    # Nested queries - multiple levels
    print("\nNested Queries (Multiple Levels):")
    queries = [
        "((temple OR mandir) AND (swami OR guru))",
        "(vishweshwara AND (temple OR (mandir AND shrine)))",
        "((swami AND anandashram) OR (swami AND parijnanashram))"
    ]
    
    for query in queries:
        tokens = tokenize_query(query)
        parsed = parse_boolean_query(query)
        print(f"Query: '{query}'")
        print(f"Tokens: {tokens}")
        print(f"Parsed: {parsed}")
        print("-" * 40)

def test_search():
    """Test the search functionality with various queries"""
    print("\n=== TESTING SEARCH FUNCTIONALITY ===")
    
    # Simple search
    print("\nSimple Search:")
    query = "temple OR mandir"
    print(f"Searching for: '{query}'")
    results = search_knowledge_base(query)
    print(f"Found {len(results)} results")
    
    # AND search
    print("\nAND Search:")
    query = "swami AND anandashram"
    print(f"Searching for: '{query}'")
    results = search_knowledge_base(query)
    print(f"Found {len(results)} results")
    
    # Nested search
    print("\nNested Search:")
    query = "(temple OR mandir) AND (swami OR guru)"
    print(f"Searching for: '{query}'")
    results = search_knowledge_base(query)
    print(f"Found {len(results)} results")
    
    # Complex nested search
    print("\nComplex Nested Search:")
    query = "((swami AND anandashram) OR (swami AND parijnanashram))"
    print(f"Searching for: '{query}'")
    results = search_knowledge_base(query)
    print(f"Found {len(results)} results")

if __name__ == "__main__":
    print("COMPREHENSIVE SEARCH ENGINE TEST")
    print("===============================")
    
    # Test the parser
    test_parser()
    
    # Test the search functionality
    test_search() 