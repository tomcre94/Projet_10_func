import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from .utils import normalize_scores, filter_read_articles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CollaborativeFilteringRecommender:
    def __init__(self, user_interactions: pd.DataFrame, articles_metadata: pd.DataFrame, config: Dict):
        self.user_interactions = user_interactions
        self.articles_metadata = articles_metadata
        self.config = config
        
        # Pre-calculate user-item matrix for efficient lookup
        self.user_article_matrix, self.user_to_idx, self.idx_to_user, \
            self.article_to_idx, self.idx_to_article = self._create_user_article_matrix()
        
        logging.info("CollaborativeFilteringRecommender initialisé.")

    def _create_user_article_matrix(self):
        """
        Crée une matrice utilisateur-article (sparse) à partir des interactions.
        """
        logging.info("Création de la matrice utilisateur-article...")
        
        # Create mappings for user_id and article_id to contiguous indices
        unique_users = self.user_interactions['user_id'].unique()
        unique_articles = self.articles_metadata['article_id'].unique() # Use all articles from metadata for consistency

        user_to_idx = {user_id: i for i, user_id in enumerate(unique_users)}
        idx_to_user = {i: user_id for user_id, i in user_to_idx.items()}

        article_to_idx = {article_id: i for i, article_id in enumerate(unique_articles)}
        idx_to_article = {i: article_id for article_id, i in article_to_idx.items()}

        # Prepare data for sparse matrix
        rows = self.user_interactions['user_id'].map(user_to_idx)
        cols = self.user_interactions['click_article_id'].map(article_to_idx)
        
        # Filter out interactions where article_id might not be in unique_articles (due to previous filtering)
        valid_rows_mask = rows.notna()
        valid_cols_mask = cols.notna()
        valid_interactions = valid_rows_mask & valid_cols_mask

        rows = rows[valid_interactions].astype(int)
        cols = cols[valid_interactions].astype(int)
        data = np.ones(len(rows)) # Implicit feedback: 1 for interaction

        user_article_matrix = csr_matrix((data, (rows, cols)), 
                                         shape=(len(unique_users), len(unique_articles)))
        
        logging.info(f"Matrice utilisateur-article créée: {user_article_matrix.shape} (sparsité: {100 * (1 - user_article_matrix.nnz / (user_article_matrix.shape[0] * user_article_matrix.shape[1])):.2f}%)")
        return user_article_matrix, user_to_idx, idx_to_user, article_to_idx, idx_to_article

    def _find_similar_users(self, user_idx: int, max_similar_users: int) -> List[int]:
        """
        Trouve les utilisateurs les plus similaires à un utilisateur donné.
        Utilise la similarité cosinus sur la matrice utilisateur-article.
        """
        if user_idx not in self.idx_to_user:
            logging.warning(f"Utilisateur avec index {user_idx} non trouvé dans le mapping.")
            return []

        user_vector = self.user_article_matrix[user_idx]
        
        # Calculate cosine similarity between the user vector and all other user vectors
        # This can be slow for very large matrices. Consider approximate methods for production.
        similarities = sk_cosine_similarity(user_vector, self.user_article_matrix).flatten()
        
        # Get indices of similar users, excluding the user itself
        similar_user_indices = similarities.argsort()[::-1][1:] # Exclude self
        
        # Filter by similarity threshold if needed (not explicitly in prompt, but good practice)
        # similarities[similar_user_indices] > self.config['similarity_threshold']
        
        # Take top N similar users
        top_similar_users = [idx for idx in similar_user_indices if similarities[idx] > 0][:max_similar_users] # Only positive similarity
        
        logging.info(f"Trouvé {len(top_similar_users)} utilisateurs similaires pour l'utilisateur {self.idx_to_user[user_idx]}.")
        return top_similar_users

    def recommend(self, user_id: int, n_recommendations: int = 5) -> Dict[int, float]:
        """
        Recommande des articles basés sur le filtrage collaboratif (user-based).
        
        Args:
            user_id: ID de l'utilisateur.
            n_recommendations: Nombre de recommandations à générer (avant combinaison).
            
        Returns:
            Dictionnaire des scores de recommandation {article_id: score}.
        """
        logging.info(f"Génération de recommandations basées sur le filtrage collaboratif pour l'utilisateur {user_id}.")

        user_idx = self.user_to_idx.get(user_id)
        if user_idx is None:
            logging.info(f"Utilisateur {user_id} non trouvé dans les données d'interactions. Retourne des scores vides.")
            return {}

        # Find similar users
        max_similar_users = self.config['max_similar_users']
        similar_user_indices = self._find_similar_users(user_idx, max_similar_users)

        if not similar_user_indices:
            logging.info(f"Aucun utilisateur similaire trouvé pour l'utilisateur {user_id}. Retourne des scores vides.")
            return {}

        # Collect articles read by similar users
        candidate_article_scores = {}
        for s_user_idx in similar_user_indices:
            # Get articles read by similar user
            articles_read_by_similar_user_indices = self.user_article_matrix[s_user_idx].nonzero()[1]
            
            for article_idx in articles_read_by_similar_user_indices:
                article_id = self.idx_to_article[article_idx]
                # Simple scoring: increment count for each time an article is read by a similar user
                # Can be weighted by similarity score of the similar user
                candidate_article_scores[article_id] = candidate_article_scores.get(article_id, 0) + 1 
        
        # Filter out articles already read by the current user
        user_read_article_ids = self.user_interactions[self.user_interactions['user_id'] == user_id]['click_article_id'].unique()
        filtered_candidate_scores = {
            aid: score for aid, score in candidate_article_scores.items() 
            if aid not in user_read_article_ids
        }

        # Normalize scores
        normalized_scores = normalize_scores(filtered_candidate_scores)
        
        logging.info(f"Recommandations collaboratives générées pour l'utilisateur {user_id}.")
        return normalized_scores
