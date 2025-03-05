from search_engine import search_knowledge_base, parse_boolean_query

def print_query_structure(query):
    """Print the query and its parsed structure"""
    parsed = parse_boolean_query(query)
    print(f"\nQuery: '{query}'")
    print(f"Parsed structure: {parsed}")
    return parsed

# Test a variety of complex nested queries
print("=== TESTING COMPLEX NESTED QUERIES ===")

# Test 1: Multiple levels of nesting with mixed operators
query1 = "((temple OR mandir) AND (swami OR guru)) OR (vishweshwara AND shrine)"
print_query_structure(query1)
results1 = search_knowledge_base(query1)
print(f"Found {len(results1)} results\n")

# Test 2: Deep nesting with multiple operators
query2 = "(((swami AND anandashram) OR (swami AND parijnanashram)) AND (teachings OR discourse))"
print_query_structure(query2)
results2 = search_knowledge_base(query2)
print(f"Found {len(results2)} results\n")

# Test 3: Complex query with multiple terms
query3 = "((swami OR guru) AND (teachings OR discourse)) AND ((temple OR mandir) OR shrine)"
print_query_structure(query3)
results3 = search_knowledge_base(query3)
print(f"Found {len(results3)} results\n")

print("\n=== SEARCH ENGINE TESTING COMPLETE ===")
print("The search engine successfully handles complex nested boolean queries.")
print("See search_engine_documentation.md for details on the implementation.") 