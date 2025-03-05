from search_engine import tokenize_query, parse_boolean_query

def debug_parse(query):
    print(f"\nParsing query: '{query}'")
    tokens = tokenize_query(query)
    print(f"Tokens: {tokens}")
    parsed = parse_boolean_query(query)
    print(f"Parsed structure: {parsed}")
    return parsed

# Test simple queries
print("SIMPLE QUERIES:")
debug_parse("temple")
debug_parse("temple AND mandir")
debug_parse("temple OR mandir")

# Test one level of nesting
print("\nONE LEVEL OF NESTING:")
debug_parse("(temple OR mandir) AND swami")
debug_parse("temple AND (swami OR guru)")

# Test multiple levels of nesting
print("\nMULTIPLE LEVELS OF NESTING:")
debug_parse("(temple OR (mandir AND shrine)) AND swami")
debug_parse("((temple OR mandir) AND (swami OR guru))")
debug_parse("(vishweshwara AND (temple OR (mandir AND shrine)))")

# Test complex query
print("\nCOMPLEX QUERY:")
debug_parse("((swami AND anandashram) OR (swami AND parijnanashram))") 