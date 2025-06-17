import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import logging
from typing import List, Dict # Import List and Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_cosine_similarity(embedding_matrix: np.ndarray, vector: np.ndarray = None) -> np.ndarray:
    """
    Calcule la similarité cosinus entre un vecteur et une matrice d'embeddings,
    ou entre toutes les paires de vecteurs dans une matrice.
    
    Args:
        embedding_matrix: Matrice NumPy des embeddings (articles x dimensions).
        vector: Vecteur NumPy (1 x dimensions) pour calculer la similarité avec la matrice.
                Si None, calcule la similarité entre toutes les paires de la matrice.
                
    Returns:
        Un tableau NumPy des scores de similarité.
    """
    if vector is not None:
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        return sk_cosine_similarity(vector, embedding_matrix)[0]
    else:
        # For large matrices, this can be memory intensive.
        # Consider using approximate nearest neighbors (ANN) for production.
        return sk_cosine_similarity(embedding_matrix)

def normalize_scores(scores: dict) -> dict:
    """
    Normalise un dictionnaire de scores entre 0 et 1.
    """
    if not scores:
        return {}
    
    min_score = min(scores.values())
    max_score = max(scores.values())
    
    if max_score == min_score:
        return {k: 0.5 for k in scores} # Avoid division by zero, return neutral score
    
    normalized_scores = {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}
    return normalized_scores

def get_top_n(scores: Dict[int, float], n: int) -> List[Dict]:
    """
    Retourne les N meilleurs articles à partir d'un dictionnaire de scores.
    """
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_n_articles = [{'article_id': article_id, 'score': score} for article_id, score in sorted_scores[:n]]
    return top_n_articles

def filter_read_articles(recommendations: List[Dict], user_id: int, user_interactions: pd.DataFrame) -> List[Dict]:
    """
    Filtre les articles déjà lus par l'utilisateur des recommandations.
    """
    read_article_ids = user_interactions[user_interactions['user_id'] == user_id]['click_article_id'].unique()
    filtered_recommendations = [rec for rec in recommendations if rec['article_id'] not in read_article_ids]
    return filtered_recommendations

def ensure_diversity(recommendations: List[Dict], articles_metadata: pd.DataFrame, diversity_factor: float = 0.2) -> List[Dict]:
    """
    Assure la diversité des catégories dans les recommandations.
    Priorise les articles de catégories moins représentées si le facteur de diversité est élevé.
    Ceci est une implémentation simplifiée.
    """
    if not recommendations:
        return []

    # Map article_id to category_id
    article_to_category = articles_metadata.set_index('article_id')['category_id'].to_dict()
    
    final_recommendations = []
    seen_categories = set()
    
    # Sort by score initially
    sorted_recs = sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    for rec in sorted_recs:
        category_id = article_to_category.get(rec['article_id'])
        
        if category_id is None: # Article not found in metadata, include it but log warning
            logging.warning(f"Article {rec['article_id']} not found in metadata for diversity check.")
            final_recommendations.append(rec)
            continue

        if category_id not in seen_categories:
            final_recommendations.append(rec)
            seen_categories.add(category_id)
        else:
            # If category already seen, apply diversity factor:
            # Only add if its score is significantly higher than others in its category,
            # or if we haven't reached n_recommendations yet and need more articles.
            # This is a heuristic. A more robust approach would involve re-ranking.
            if len(final_recommendations) < len(recommendations) * (1 - diversity_factor):
                final_recommendations.append(rec)
    
    # Re-sort by score after diversity adjustment (if any)
    final_recommendations = sorted(final_recommendations, key=lambda x: x['score'], reverse=True)
    
    return final_recommendations
