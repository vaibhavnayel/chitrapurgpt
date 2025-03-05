from search_engine import search_knowledge_base, parse_boolean_query
import json

# Test a more complex nested query
query = "((swami OR guru) AND (teachings OR discourse)) AND (dharma OR spiritual)"
print(f"Searching for: {query}")
parsed_query = parse_boolean_query(query)
print(f"Parsed query structure: {json.dumps(parsed_query, indent=2)}")
search_knowledge_base(query) 