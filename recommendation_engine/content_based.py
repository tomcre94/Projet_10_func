import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from .utils import calculate_cosine_similarity, normalize_scores, filter_read_articles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentBasedRecommender:
    def __init__(self, user_interactions: pd.DataFrame, articles_metadata: pd.DataFrame, 
                 embeddings_optimized: np.ndarray, article_id_to_embedding_idx: Dict[int, int], config: Dict):
        logger.info("Initializing ContentBasedRecommender...")
        self.user_interactions = user_interactions
        self.articles_metadata = articles_metadata
        self.embeddings_optimized = embeddings_optimized
        self.article_id_to_embedding_idx = article_id_to_embedding_idx
        self.config = config
        
        logger.info("ContentBasedRecommender initialized successfully.")

    def _get_article_embedding(self, article_id: int) -> np.ndarray:
        """
        Récupère l'embedding d'un article donné.
        """
        idx = self.article_id_to_embedding_idx.get(article_id)
        if idx is not None and idx < self.embeddings_optimized.shape[0]:
            return self.embeddings_optimized[idx]
        logging.warning(f"Embedding non trouvé pour l'article_id: {article_id}")
        return None

    def recommend(self, user_id: int, n_recommendations: int = 5) -> Dict[int, float]:
        """
        Recommande des articles basés sur le contenu, similaires aux derniers articles lus par l'utilisateur.
        
        Args:
            user_id: ID de l'utilisateur.
            n_recommendations: Nombre de recommandations à générer (avant combinaison).
            
        Returns:
            Dictionnaire des scores de similarité {article_id: score}.
        """
        logging.info(f"Génération de recommandations basées sur le contenu pour l'utilisateur {user_id}.")

        user_history = self.user_interactions[self.user_interactions['user_id'] == user_id]
        
        if user_history.empty:
            logging.info(f"Aucun historique d'interactions trouvé pour l'utilisateur {user_id}. Retourne des scores vides.")
            return {}

        # Récupérer les 5 derniers articles lus par l'utilisateur
        # Sort by timestamp to get the latest articles
        latest_articles = user_history.sort_values(by='click_timestamp', ascending=False)['click_article_id'].head(5).tolist()
        
        user_profile_embeddings = []
        for article_id in latest_articles:
            embedding = self._get_article_embedding(article_id)
            if embedding is not None:
                user_profile_embeddings.append(embedding)
        
        if not user_profile_embeddings:
            logging.info(f"Aucun embedding valide trouvé pour les derniers articles lus par l'utilisateur {user_id}. Retourne des scores vides.")
            return {}

        # Calculer le centroïde des embeddings de ces articles
        user_profile_centroid = np.mean(user_profile_embeddings, axis=0)

        # Trouver les articles les plus similaires au centroïde
        # Calculer la similarité avec tous les articles disponibles (non lus)
        
        # Get all article IDs that are not in the user's read history
        read_article_ids = user_history['click_article_id'].unique()
        
        candidate_article_ids = [
            aid for aid in self.articles_metadata['article_id'].tolist() 
            if aid not in read_article_ids and self._get_article_embedding(aid) is not None
        ]
        
        if not candidate_article_ids:
            logging.info(f"Aucun article candidat disponible pour l'utilisateur {user_id} après filtrage. Retourne des scores vides.")
            return {}

        # Get embeddings for candidate articles
        candidate_embeddings = np.array([self._get_article_embedding(aid) for aid in candidate_article_ids])
        
        # Calculate cosine similarity between user centroid and candidate articles
        similarity_scores = calculate_cosine_similarity(candidate_embeddings, user_profile_centroid)
        
        # Map scores back to article_ids
        article_scores = {candidate_article_ids[i]: score for i, score in enumerate(similarity_scores)}
        
        # Normalize scores to 0-1 range
        normalized_article_scores = normalize_scores(article_scores)

        logging.info(f"Recommandations basées sur le contenu générées pour l'utilisateur {user_id}.")
        return normalized_article_scores
