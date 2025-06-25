import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict
from .data_loader import DataLoader
from .popularity_based import PopularityBasedRecommender
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender # Import collaborative filtering
from .utils import normalize_scores, filter_read_articles, ensure_diversity # Import utilities
from config import RECOMMENDATION_CONFIG # Assuming config.py is in the root directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RecommendationEngine:
    def __init__(self, data_path="processed_data/"):
        """
        Initialise le système de recommandation
        - Charge les données préparées
        - Initialise les 3 composants de recommandation
        - Pré-calcule les matrices/structures nécessaires
        """
        logging.info("Initialisation de RecommendationEngine...")
        self.data_loader = DataLoader(data_path)
        self.data_loader.load_all_data()

        self.user_interactions = self.data_loader.get_user_interactions()
        self.articles_metadata = self.data_loader.get_articles_metadata()
        self.embeddings_optimized = self.data_loader.get_embeddings_optimized()
        self.data_summary = self.data_loader.get_data_summary()

        self.config = RECOMMENDATION_CONFIG

        # Pré-calculs (à implémenter dans les composants ou ici si global)
        # Par exemple, un mapping article_id -> index d'embedding
        self.article_id_to_embedding_idx = {
            article_id: idx for idx, article_id in enumerate(self.articles_metadata['article_id'].tolist())
        }
        # Inverse mapping for convenience
        self.embedding_idx_to_article_id = {
            idx: article_id for article_id, idx in self.article_id_to_embedding_idx.items()
        }

        # Initialisation des composants de recommandation (placeholders for now)
        self.content_based_recommender = ContentBasedRecommender(
            self.user_interactions, self.articles_metadata, 
            self.embeddings_optimized, self.article_id_to_embedding_idx, self.config
        )
        self.collaborative_recommender = CollaborativeFilteringRecommender( # Initialize collaborative recommender
            self.user_interactions, self.articles_metadata, self.config
        )
        self.popularity_recommender = PopularityBasedRecommender(
            self.user_interactions, self.articles_metadata, self.config
        )

        logging.info("RecommendationEngine initialisé.")
    
    def recommend_articles(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Fonction principale de recommandation
        
        Args:
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations (défaut: 5)
            
        Returns:
            Liste de dictionnaires avec:
            - article_id
            - title (si disponible)
            - category_id  
            - score (score de recommandation)
            - reason (pourquoi cet article est recommandé)
        """
        logging.info(f"Génération de recommandations pour l'utilisateur {user_id}...")

        # Gérer le Cold Start Problem
        user_interactions_count = self.user_interactions[self.user_interactions['user_id'] == user_id].shape[0]
        
        # Handle Cold Start Problem
        user_interactions_count = self.user_interactions[self.user_interactions['user_id'] == user_id].shape[0]
        
        if user_interactions_count < self.config['min_interactions_collab']: # Cold start user (<3 interactions)
            logging.info(f"Utilisateur {user_id} en cold start ({user_interactions_count} interactions). Applique la stratégie cold start.")
            return self.popularity_recommender.recommend(user_id, n_recommendations, is_cold_start=True)
        
        # Handle users with little history (3-10 interactions)
        current_weights = self.config['weights'].copy()
        if 3 <= user_interactions_count <= 10:
            logging.info(f"Utilisateur {user_id} avec peu d'historique ({user_interactions_count} interactions). Ajuste les poids.")
            current_weights['collaborative'] = 0.15
            current_weights['content_based'] = 0.45
            current_weights['popularity'] = 0.40
        else:
            logging.info(f"Utilisateur {user_id} avec historique suffisant ({user_interactions_count} interactions). Utilise les poids par défaut.")

        # Get scores from individual recommenders
        content_scores = self.content_based_recommender.recommend(user_id, n_recommendations * 2) # Get more for combining
        collab_scores = self.collaborative_recommender.recommend(user_id, n_recommendations * 2) # Get more for combining
        popularity_scores_list = self.popularity_recommender.recommend(user_id, n_recommendations * 2) # Get more for combining
        popularity_scores = {rec['article_id']: rec['score'] for rec in popularity_scores_list}

        # Combine scores
        final_scores = self._combine_scores(content_scores, collab_scores, popularity_scores, current_weights)

        # Filter out already read articles and unavailable articles
        user_read_article_ids = self.user_interactions[self.user_interactions['user_id'] == user_id]['click_article_id'].unique()
        available_article_ids = self.articles_metadata['article_id'].unique()
        
        sorted_final_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        
        recommendations = []
        seen_article_ids_for_dedup = set(user_read_article_ids)
        
        for article_id, score in sorted_final_scores:
            if article_id not in seen_article_ids_for_dedup and article_id in available_article_ids:
                article_meta = self.articles_metadata[self.articles_metadata['article_id'] == article_id]
                if not article_meta.empty:
                    article_meta = article_meta.iloc[0]
                    recommendations.append({
                        'article_id': article_id,
                        'title': f"Article {article_id}", # Placeholder for title, assuming no title in metadata
                        'category_id': article_meta['category_id'],
                        'score': score,
                        'reason': "Combinaison hybride"
                    })
                    seen_article_ids_for_dedup.add(article_id)
                    if len(recommendations) >= n_recommendations:
                        break
                else:
                    logging.warning(f"Article {article_id} not found in metadata during final recommendation assembly.")
        
        # Ensure diversity
        recommendations = ensure_diversity(recommendations, self.articles_metadata, self.config['category_diversity_factor'])
        recommendations = recommendations[:n_recommendations] # Trim again after diversity

        logging.info(f"Recommandations générées pour l'utilisateur {user_id}.")
        return recommendations
        
    def _combine_scores(self, content_scores: Dict[int, float], collab_scores: Dict[int, float], popularity_scores: Dict[int, float], weights: Dict[str, float]) -> Dict[int, float]:
        """
        Combine les scores des 3 approches avec pondération.
        """
        logging.info("Combinaison des scores des différentes approches...")
        
        all_article_ids = set(content_scores.keys()) | set(collab_scores.keys()) | set(popularity_scores.keys())
        
        final_scores = {}

        # Collect all scores for normalization
        all_content_scores = list(content_scores.values())
        all_collab_scores = list(collab_scores.values())
        all_popularity_scores = list(popularity_scores.values())

        # Normalize scores from each component to 0-1 range
        # Use a global min/max for each component if possible, or per-call if dynamic
        # For simplicity, normalize based on the scores present in this call
        normalized_content_scores = normalize_scores(content_scores)
        normalized_collab_scores = normalize_scores(collab_scores)
        normalized_popularity_scores = normalize_scores(popularity_scores)
        
        for article_id in all_article_ids:
            c_score_norm = normalized_content_scores.get(article_id, 0.0)
            col_score_norm = normalized_collab_scores.get(article_id, 0.0)
            pop_score_norm = normalized_popularity_scores.get(article_id, 0.0)
            
            combined_score = (c_score_norm * weights['content_based'] +
                              col_score_norm * weights['collaborative'] +
                              pop_score_norm * weights['popularity'])
            final_scores[article_id] = combined_score
            
        logging.info("Scores combinés.")
        return final_scores

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Ensure processed_data exists and contains the necessary files
    # You might need to run data_preparation.py first
    
    # Create dummy processed_data for testing if not available
    if not os.path.exists("processed_data/user_interactions.json"):
        print("Creating dummy processed_data for testing...")
        os.makedirs("processed_data", exist_ok=True)
        
        dummy_clicks = pd.DataFrame({
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 10001, 10001],
            'session_id': [101, 101, 102, 201, 202, 301, 301, 302, 303, 401, 402],
            'click_article_id': [10, 11, 12, 10, 13, 11, 14, 15, 16, 10, 11],
            'click_timestamp': [1678886400000, 1678886500000, 1678886600000, 1678886700000, 1678886800000, 1678886900000, 1678887000000, 1678887100000, 1678887200000, 1678887300000, 1678887400000]
        })
        dummy_clicks.to_json("processed_data/user_interactions.json", orient='records', lines=True, date_unit='ms')

        dummy_meta = pd.DataFrame({
            'article_id': [10, 11, 12, 13, 14, 15, 16],
            'category_id': [1, 2, 1, 3, 2, 1, 3],
            'created_at_ts': [1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000, 1678886000000],
            'publisher_id': [1, 1, 2, 2, 1, 1, 2],
            'words_count': [100, 120, 150, 110, 130, 140, 160]
        })
        dummy_meta.to_json("processed_data/articles_metadata.json", orient='records', lines=True, date_unit='ms')

        dummy_embeddings = np.random.rand(7, 52).astype(np.float32) # 7 articles, 52 dimensions
        with open("processed_data/embeddings_optimized.pkl", 'wb') as f:
            pickle.dump(dummy_embeddings, f)

        dummy_summary = {
            "total_interactions": len(dummy_clicks),
            "total_users": dummy_clicks['user_id'].nunique(),
            "total_articles": dummy_clicks['click_article_id'].nunique(),
            "total_sessions": dummy_clicks['session_id'].nunique(),
            "embedding_dimensions": dummy_embeddings.shape[1]
        }
        with open("processed_data/data_summary.json", 'w') as f:
            json.dump(dummy_summary, f, indent=4)
        print("Dummy data created.")

    recommender = RecommendationEngine()
    
    # Test with a user who has interactions
    user_id_test = 1
    recommendations = recommender.recommend_articles(user_id_test)
    print(f"\nRecommandations pour l'utilisateur {user_id_test}:")
    for rec in recommendations:
        print(rec)

    # Test with a cold start user (e.g., user_id 10001 has only 2 interactions in dummy data)
    user_id_cold_start = 10001
    recommendations_cold_start = recommender.recommend_articles(user_id_cold_start)
    print(f"\nRecommandations pour l'utilisateur cold start {user_id_cold_start}:")
    for rec in recommendations_cold_start:
        print(rec)
