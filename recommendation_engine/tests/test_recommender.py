import unittest
import os
import pandas as pd
import numpy as np
import json
import time
import pickle # Import pickle
import shutil # Import shutil for rmtree
from recommendation_engine.recommender import RecommendationEngine
from config import RECOMMENDATION_CONFIG

# Helper function to create dummy processed_data for testing
def create_dummy_processed_data(processed_data_path="processed_data_test/"):
    # Ensure the directory exists and is empty for a clean test environment
    if os.path.exists(processed_data_path):
        import shutil
        shutil.rmtree(processed_data_path)
    os.makedirs(processed_data_path, exist_ok=True)
        
    dummy_clicks = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 10001, 10001, 10002], # User 10001 has 2 interactions (cold start), 10002 has 0
        'session_id': [101, 101, 102, 201, 202, 301, 301, 302, 303, 401, 402, 501],
        'click_article_id': [10, 11, 12, 10, 13, 11, 14, 15, 16, 10, 11, 17],
        'click_timestamp': [1678886400000, 1678886500000, 1678886600000, 1678886700000, 1678886800000, 1678886900000, 1678887000000, 1678887100000, 1678887200000, 1678887300000, 1678887400000, 1678887500000]
    })
    dummy_clicks.to_json(os.path.join(processed_data_path, "user_interactions.json"), orient='records', lines=True, date_unit='ms')

    dummy_meta = pd.DataFrame({
        'article_id': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], # Add more articles for diversity
        'category_id': [1, 2, 1, 3, 2, 1, 3, 4, 5, 6, 7],
        'created_at_ts': [1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000],
        'publisher_id': [1, 1, 2, 2, 1, 1, 2, 3, 3, 4, 4],
        'words_count': [100, 120, 150, 110, 130, 140, 160, 170, 180, 190, 200]
    })
    dummy_meta.to_json(os.path.join(processed_data_path, "articles_metadata.json"), orient='records', lines=True, date_unit='ms')

    dummy_embeddings = np.random.rand(len(dummy_meta), 52).astype(np.float32) # Match articles in dummy_meta
    with open(os.path.join(processed_data_path, "embeddings_optimized.pkl"), 'wb') as f:
        pickle.dump(dummy_embeddings, f)

    dummy_summary = {
        "total_interactions": len(dummy_clicks),
        "total_users": dummy_clicks['user_id'].nunique(),
        "total_articles": dummy_clicks['click_article_id'].nunique(),
        "total_sessions": dummy_clicks['session_id'].nunique(),
        "embedding_dimensions": dummy_embeddings.shape[1]
    }
    with open(os.path.join(processed_data_path, "data_summary.json"), 'w') as f:
        json.dump(dummy_summary, f, indent=4)
    print(f"Dummy data created for testing in {processed_data_path}.")
    return processed_data_path # Return the path

class TestRecommendationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create dummy data once for all tests in a dedicated test directory
        cls.test_data_path = create_dummy_processed_data()
        cls.recommender = RecommendationEngine(data_path=cls.test_data_path)

    def test_initialization(self):
        self.assertIsNotNone(self.recommender.user_interactions)
        self.assertIsNotNone(self.recommender.articles_metadata)
        self.assertIsNotNone(self.recommender.embeddings_optimized)
        self.assertIsNotNone(self.recommender.data_summary)
        self.assertIsNotNone(self.recommender.content_based_recommender)
        self.assertIsNotNone(self.recommender.popularity_recommender)
        self.assertIsNotNone(self.recommender.collaborative_recommender)
        self.assertGreater(len(self.recommender.article_id_to_embedding_idx), 0)

    def test_recommend_articles_existing_user(self):
        # User 1 has 3 interactions (10, 11, 12)
        user_id = 1
        n_recs = 5
        recommendations = self.recommender.recommend_articles(user_id, n_recs)
        
        self.assertEqual(len(recommendations), n_recs)
        self.assertIsInstance(recommendations, list)
        self.assertIsInstance(recommendations[0], dict)
        self.assertIn('article_id', recommendations[0])
        self.assertIn('score', recommendations[0])
        self.assertIn('reason', recommendations[0])
        
        # Check for deduplication: articles 10, 11, 12 should not be in recommendations
        read_articles = self.recommender.user_interactions[self.recommender.user_interactions['user_id'] == user_id]['click_article_id'].unique()
        for rec in recommendations:
            self.assertNotIn(rec['article_id'], read_articles)

    def test_recommend_articles_cold_start_user(self):
        # User 10001 has 2 interactions (cold start)
        user_id = 10001
        n_recs = 5
        recommendations = self.recommender.recommend_articles(user_id, n_recs)
        
        self.assertEqual(len(recommendations), n_recs)
        self.assertIsInstance(recommendations, list)
        self.assertIsInstance(recommendations[0], dict)
        self.assertEqual(recommendations[0]['reason'], "Popularité/Tendance") # Should be popularity-based
        
        # Check for deduplication: articles 10, 11 should not be in recommendations
        read_articles = self.recommender.user_interactions[self.recommender.user_interactions['user_id'] == user_id]['click_article_id'].unique()
        for rec in recommendations:
            self.assertNotIn(rec['article_id'], read_articles)

    def test_recommend_articles_non_existent_user(self):
        # User 99999 does not exist in dummy data
        user_id = 99999
        n_recs = 5
        recommendations = self.recommender.recommend_articles(user_id, n_recs)
        
        self.assertEqual(len(recommendations), n_recs)
        self.assertIsInstance(recommendations, list)
        self.assertIsInstance(recommendations[0], dict)
        self.assertEqual(recommendations[0]['reason'], "Popularité/Tendance") # Should be popularity-based (cold start path)

    def test_performance(self):
        user_id = 1 # An existing user
        n_recs = 5
        start_time = time.time()
        self.recommender.recommend_articles(user_id, n_recs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nPerformance test for user {user_id}: {execution_time:.4f} seconds")
        self.assertLess(execution_time, 30, "Recommendation took longer than 30 seconds (Azure Functions limit)")

    def test_diversity(self):
        # This test is more conceptual for dummy data, real diversity needs more complex data
        user_id = 1
        n_recs = 5
        recommendations = self.recommender.recommend_articles(user_id, n_recs)
        
        categories = [rec['category_id'] for rec in recommendations]
        # With dummy data, it's hard to guarantee diversity without specific article setups
        # For now, just check if there are at least 2 unique categories if n_recs > 1
        if n_recs > 1:
            self.assertGreaterEqual(len(set(categories)), min(n_recs, 2)) # At least 2 unique categories if possible

if __name__ == '__main__':
    unittest.main()
