import unittest
from search_engine import tokenize_query, parse_boolean_query, evaluate_query, search_knowledge_base
from retrievers import Document
import json

class TestSearchEngine(unittest.TestCase):
    def test_tokenize_query(self):
        query = "(temple OR mandir) AND (vishweshwara OR viswesvara)"
        tokens = tokenize_query(query)
        expected = ['(', 'temple', 'OR', 'mandir', ')', 'AND', '(', 'vishweshwara', 'OR', 'viswesvara', ')']
        self.assertEqual(tokens, expected)
    
    def test_parse_boolean_query_simple(self):
        query = "temple AND mandir"
        parsed = parse_boolean_query(query)
        expected = {'AND': ['temple', 'mandir']}
        self.assertEqual(parsed, expected)
    
    def test_parse_boolean_query_nested(self):
        query = "(temple OR mandir) AND (vishweshwara OR viswesvara)"
        parsed = parse_boolean_query(query)
        print(f"DEBUG - Parsed query: {json.dumps(parsed, indent=2)}")
        expected = {'AND': [{'OR': ['temple', 'mandir']}, {'OR': ['vishweshwara', 'viswesvara']}]}
        self.assertEqual(parsed, expected)
    
    def test_evaluate_query(self):
        # Create a test document
        doc = Document(
            page_content="The Vishweshwara temple in Benares is a famous Hindu mandir.",
            metadata={"title": "Temples of India", "source": "test"}
        )
        
        # Test simple term
        self.assertTrue(evaluate_query("temple", doc))
        self.assertFalse(evaluate_query("mosque", doc))
        
        # Test AND operation
        self.assertTrue(evaluate_query({'AND': ['temple', 'mandir']}, doc))
        self.assertFalse(evaluate_query({'AND': ['temple', 'mosque']}, doc))
        
        # Test OR operation
        self.assertTrue(evaluate_query({'OR': ['temple', 'mosque']}, doc))
        
        # Test nested query
        nested_query = {'AND': [{'OR': ['temple', 'mandir']}, {'OR': ['Vishweshwara', 'Viswesvara']}]}
        self.assertTrue(evaluate_query(nested_query, doc))

if __name__ == "__main__":
    # Run the tests
    unittest.main()
    
    # Also perform a real search on the knowledge base
    print("\n\nTesting with real knowledge base:")
    query = "(viswesvara OR vishweshwara OR visweswara) AND (temple OR mandir)"
    print(f"Searching for: {query}")
    search_knowledge_base(query) 