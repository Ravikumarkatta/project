import unittest
import json
import os
from app import app

class TestBibleAPI(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
    def test_home_endpoint(self):
        response = self.client.get('/')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('message', data)
        self.assertIn('endpoints', data)
        
    def test_get_verse(self):
        response = self.client.get('/api/v1/verse/Genesis/1/1')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['book'], 'Genesis')
        self.assertEqual(data['chapter'], 1)
        self.assertEqual(data['verse'], 1)
        self.assertIn('text', data)
        
    def test_get_chapter(self):
        response = self.client.get('/api/v1/chapter/Genesis/1')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['book'], 'Genesis')
        self.assertEqual(data['chapter'], 1)
        self.assertIn('verses', data)
        
    def test_search_bible(self):
        # Test with query
        response = self.client.get('/api/v1/search?q=beginning')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', data)
        self.assertIn('count', data)
        self.assertIn('query', data)
        
        # Test with empty query
        response = self.client.get('/api/v1/search')
        self.assertEqual(response.status_code, 400)
        
    def test_list_books(self):
        response = self.client.get('/api/v1/books')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('books', data)
        self.assertIn('count', data)
        self.assertGreater(len(data['books']), 0)

if __name__ == '__main__':
    unittest.main()