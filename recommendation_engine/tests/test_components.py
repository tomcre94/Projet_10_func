import unittest
import pandas as pd
import numpy as np
import os
import json
from recommendation_engine.content_based import ContentBasedRecommender
from recommendation_engine.popularity_based import PopularityBasedRecommender
from recommendation_engine.collaborative_filtering import CollaborativeFilteringRecommender
from config import RECOMMENDATION_CONFIG

# Helper function to create dummy processed_data for testing
import pickle # Import pickle
import shutil # Import shutil for rmtree

# Helper function to create dummy processed_data for testing
def create_dummy_processed_data_for_components(processed_data_path="processed_data_components_test/"):
    # Ensure the directory exists and is empty for a clean test environment
    if os.path.exists(processed_data_path):
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
    print(f"Dummy data created for component testing in {processed_data_path}.")
    return processed_data_path # Return the path

class TestRecommendationComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = create_dummy_processed_data_for_components()
        processed_data_path = cls.test_data_path # Use the returned path
        cls.user_interactions = pd.read_json(os.path.join(processed_data_path, 'user_interactions.json'), lines=True)
        cls.articles_metadata = pd.read_json(os.path.join(processed_data_path, 'articles_metadata.json'), lines=True)
        with open(os.path.join(processed_data_path, 'embeddings_optimized.pkl'), 'rb') as f:
            cls.embeddings_optimized = pickle.load(f)
        cls.config = RECOMMENDATION_CONFIG
        
        cls.article_id_to_embedding_idx = {
            article_id: idx for idx, article_id in enumerate(cls.articles_metadata['article_id'].tolist())
        }

    def test_popularity_based_recommender(self):
        recommender = PopularityBasedRecommender(self.user_interactions, self.articles_metadata, self.config)
        user_id = 1 # Any user, as popularity is global
        n_recs = 5
        recommendations = recommender.recommend(user_id, n_recs)
        
        self.assertEqual(len(recommendations), n_recs)
        self.assertIsInstance(recommendations[0], dict)
        self.assertIn('article_id', recommendations[0])
        self.assertIn('score', recommendations[0])
        self.assertIn('reason', recommendations[0])
        self.assertEqual(recommendations[0]['reason'], "Popularité/Tendance")

        # Check if scores are normalized (roughly between 0 and 1)
        for rec in recommendations:
            self.assertGreaterEqual(rec['score'], 0)
            self.assertLessEqual(rec['score'], 1)

    def test_content_based_recommender(self):
        recommender = ContentBasedRecommender(self.user_interactions, self.articles_metadata, 
                                              self.embeddings_optimized, self.article_id_to_embedding_idx, self.config)
        user_id = 1 # User with history: 10, 11, 12
        n_recs = 5
        scores = recommender.recommend(user_id, n_recs)
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0) # Should return some scores
        
        # Check if scores are normalized
        for score in scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        
        # Check deduplication (articles 10, 11, 12 should not be in the scores)
        self.assertNotIn(10, scores)
        self.assertNotIn(11, scores)
        self.assertNotIn(12, scores)

    def test_collaborative_filtering_recommender(self):
        recommender = CollaborativeFilteringRecommender(self.user_interactions, self.articles_metadata, self.config)
        user_id = 1 # User with history
        n_recs = 5
        scores = recommender.recommend(user_id, n_recs)
        
        self.assertIsInstance(scores, dict)
        # With dummy data, it's hard to guarantee similar users and non-read articles
        # So, just check if it returns a dictionary of scores
        
        # Check if scores are normalized
        for score in scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        
        # Check deduplication (articles 10, 11, 12 should not be in the scores)
        self.assertNotIn(10, scores)
        self.assertNotIn(11, scores)
        self.assertNotIn(12, scores)

if __name__ == '__main__':
    unittest.main()
