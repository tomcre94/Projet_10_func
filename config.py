RECOMMENDATION_CONFIG = {
    'weights': {
        'content_based': 0.4,
        'collaborative': 0.3, 
        'popularity': 0.3
    },
    'cold_start_weights': {
        'popularity': 0.7,
        'category_diversity': 0.3
    },
    'similarity_threshold': 0.1,
    'min_interactions_collab': 3,
    'max_similar_users': 50,
    'freshness_decay_days': 7,
    'category_diversity_factor': 0.2
}
