import unittest
from unittest.mock import MagicMock, patch
import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.mongodb import get_database, insert_message_mongo_db, get_all_data_from_mongo_db

class TestMongoDB(unittest.TestCase):

    @patch('src.mongodb.MongoClient')
    @patch('src.mongodb.CONNECTION_STRING', 'mongodb://localhost:27017')
    def test_get_database(self, mock_client):
        db = get_database()
        mock_client.assert_called_once_with('mongodb://localhost:27017')
        self.assertEqual(db, mock_client.return_value['icaro'])

    @patch('src.mongodb.get_database')
    def test_insert_message_mongo_db(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection

        asyncio.run(insert_message_mongo_db("test_title", "test_message", True))

        mock_collection.insert_one.assert_called_once()
        args, _ = mock_collection.insert_one.call_args
        self.assertEqual(args[0]['title'], "test_title")
        self.assertEqual(args[0]['message'], "test_message")
        self.assertEqual(args[0]['alert'], True)
        self.assertIn('timestamp', args[0])

    @patch('src.mongodb.get_database')
    def test_get_all_data_from_mongo_db(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_collection = MagicMock()
        mock_db.__getitem__.return_value = mock_collection
        
        mock_collection.find.return_value = [
            {'_id': '123', 'title': 't1', 'message': 'm1', 'alert': True},
            {'_id': '456', 'title': 't2', 'message': 'm2', 'alert': False}
        ]

        result = get_all_data_from_mongo_db()
        self.assertEqual(len(result['alerts']), 2)
        self.assertEqual(result['alerts'][0]['_id'], '123')
        self.assertEqual(result['alerts'][1]['title'], 't2')

if __name__ == "__main__":
    unittest.main()
