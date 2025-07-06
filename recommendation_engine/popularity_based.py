import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from .utils import normalize_scores, get_top_n, ensure_diversity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PopularityBasedRecommender:
    def __init__(self, user_interactions: pd.DataFrame, articles_metadata: pd.DataFrame, config: Dict):
        logger.info("Initializing PopularityBasedRecommender...")
        self.user_interactions = user_interactions
        self.articles_metadata = articles_metadata
        self.config = config
        
        self.article_popularity_scores = self._calculate_global_popularity()
        self.category_popularity_scores = self._calculate_category_popularity()
        
        logger.info("PopularityBasedRecommender initialized successfully.")

    def _calculate_global_popularity(self) -> Dict[int, float]:
        """
        Calcule la popularité globale des articles basée sur le nombre de vues.
        """
        logging.info("Calcul de la popularité globale des articles...")
        article_views = self.user_interactions['click_article_id'].value_counts()
        # Normalize scores to 0-1 range
        scores = normalize_scores(article_views.to_dict())
        logging.info(f"Popularité globale calculée pour {len(scores)} articles.")
        return scores

    def _calculate_category_popularity(self) -> Dict[int, float]:
        """
        Calcule la popularité des catégories basée sur le nombre total de vues des articles dans chaque catégorie.
        """
        logging.info("Calcul de la popularité des catégories...")
        # Merge interactions with metadata to get category_id for each click
        clicks_with_category = self.user_interactions.merge(
            self.articles_metadata[['article_id', 'category_id']],
            left_on='click_article_id',
            right_on='article_id',
            how='left'
        )
        category_views = clicks_with_category['category_id'].value_counts()
        scores = normalize_scores(category_views.to_dict())
        logging.info(f"Popularité des catégories calculée pour {len(scores)} catégories.")
        return scores

    def get_popularity_scores(self, article_ids: List[int] = None) -> Dict[int, float]:
        """
        Retourne les scores de popularité globale pour les articles donnés ou pour tous les articles.
        """
        if article_ids is None:
            return self.article_popularity_scores
        else:
            return {aid: self.article_popularity_scores.get(aid, 0.0) for aid in article_ids}

    def recommend(self, user_id: int, n_recommendations: int = 5, is_cold_start: bool = False) -> List[Dict]:
        """
        Recommande des articles basés sur la popularité et la fraîcheur.
        Gère la stratégie cold start.
        
        Args:
            user_id: ID de l'utilisateur (utilisé pour le filtrage des articles déjà lus).
            n_recommendations: Nombre de recommandations.
            is_cold_start: Booléen indiquant si l'utilisateur est en cold start.
            
        Returns:
            Liste de dictionnaires avec les articles recommandés et leurs scores.
        """
        logging.info(f"Génération de recommandations basées sur la popularité pour l'utilisateur {user_id} (cold start: {is_cold_start}).")

        # Start with all popular articles
        candidate_articles = self.articles_metadata.copy()
        
        # Calculate freshness score
        # Convert 'created_at_ts' to datetime
        candidate_articles['created_at_dt'] = pd.to_datetime(candidate_articles['created_at_ts'], unit='ms')
        
        # Assume current time is the latest click timestamp for dynamic freshness, or a fixed point
        # For simplicity, let's use the max creation timestamp in the dataset as a reference point for freshness
        max_created_ts = self.articles_metadata['created_at_ts'].max()
        max_created_dt = pd.to_datetime(max_created_ts, unit='ms')

        # Calculate age in days
        candidate_articles['age_days'] = (max_created_dt - candidate_articles['created_at_dt']).dt.days

        # Apply freshness decay: score = exp(-age_days / freshness_decay_days)
        freshness_decay_days = self.config['freshness_decay_days']
        candidate_articles['freshness_score'] = np.exp(-candidate_articles['age_days'] / freshness_decay_days)
        
        # Combine popularity and freshness
        # Ensure article_id is in article_popularity_scores
        candidate_articles['popularity_score'] = candidate_articles['article_id'].map(self.article_popularity_scores).fillna(0.0)
        
        # Simple combination: popularity * freshness. Can be more complex.
        # For cold start, popularity weight is higher.
        if is_cold_start:
            # For cold start, use cold_start_weights
            popularity_weight = self.config['cold_start_weights']['popularity']
            # Category diversity will be handled by ensure_diversity utility
            
            # For cold start, we might want to prioritize articles that are popular overall
            # and then apply freshness.
            # The score here is a blend of popularity and freshness.
            candidate_articles['final_score'] = (candidate_articles['popularity_score'] * popularity_weight + 
                                                 candidate_articles['freshness_score'] * (1 - popularity_weight))
        else:
            # For regular popularity, use the general popularity weight from main config
            # This component's score will be combined later in RecommendationEngine
            candidate_articles['final_score'] = (candidate_articles['popularity_score'] * 0.7 + # Heuristic for now
                                                 candidate_articles['freshness_score'] * 0.3) # Heuristic for now
            # These weights will be overridden by the main combiner, but this gives a base score for this component

        # Filter out articles already read by the user
        read_article_ids = self.user_interactions[self.user_interactions['user_id'] == user_id]['click_article_id'].unique()
        candidate_articles = candidate_articles[~candidate_articles['article_id'].isin(read_article_ids)]

        # Sort by final score
        candidate_articles = candidate_articles.sort_values(by='final_score', ascending=False)

        # Prepare recommendations list
        recommendations = []
        for _, row in candidate_articles.head(n_recommendations * 5).iterrows(): # Take more to allow for diversity filtering
            recommendations.append({
                'article_id': row['article_id'],
                'title': f"Article {row['article_id']}", # Placeholder for title
                'category_id': row['category_id'],
                'score': row['final_score'],
                'reason': "Popularité/Tendance"
            })
        
        # Ensure diversity for cold start or if specified
        if is_cold_start: # Apply diversity for cold start
            recommendations = ensure_diversity(recommendations, self.articles_metadata, self.config['cold_start_weights']['category_diversity'])
        else: # Apply general diversity factor
            recommendations = ensure_diversity(recommendations, self.articles_metadata, self.config['category_diversity_factor'])

        # Trim to n_recommendations
        recommendations = recommendations[:n_recommendations]

        logging.info(f"Recommandations basées sur la popularité générées pour l'utilisateur {user_id}.")
        return recommendations
