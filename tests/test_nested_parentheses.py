from search_engine import search_knowledge_base, parse_boolean_query

print("Testing nested parentheses parsing:")
query1 = "((temple OR mandir) AND (swami OR guru))"
parsed1 = parse_boolean_query(query1)
print(f"Query: {query1}")
print(f"Parsed structure: {parsed1}")

query2 = "(vishweshwara AND (temple OR (mandir AND shrine)))"
parsed2 = parse_boolean_query(query2)
print(f"Query: {query2}")
print(f"Parsed structure: {parsed2}")

query3 = "((swami AND anandashram) OR (swami AND parijnanashram))"
parsed3 = parse_boolean_query(query3)
print(f"Query: {query3}")
print(f"Parsed structure: {parsed3}")

print("\nTesting search with nested parentheses:")
print(f"Searching for: {query1}")
results1 = search_knowledge_base(query1)
print(f"Found {len(results1)} results\n")

print(f"Searching for: {query3}")
results3 = search_knowledge_base(query3)
print(f"Found {len(results3)} results\n") 