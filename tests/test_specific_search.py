from search_engine import search_knowledge_base

# Try a more specific query based on the knowledge base content
query = "swami AND anandashram"
print(f"Searching for: {query}")
search_knowledge_base(query)

# Try a nested query
query = "(swami OR guru) AND (teachings OR discourse)"
print(f"\nSearching for: {query}")
search_knowledge_base(query) 