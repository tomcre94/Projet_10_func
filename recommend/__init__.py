import logging
import json
import os
import sys
import pickle
import gc
from typing import Optional, Dict, Any

import azure.functions as func
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
logger.info(f"sys.path: {sys.path}")

# Import conditionnel des modules
try:
    from recommendation_engine.recommender import RecommendationEngine
    RECOMMENDATION_MODULES_AVAILABLE = True
    logger.info("Recommendation modules imported successfully")
except ImportError as e:
    logger.error(f"Recommendation modules not available: {e}")
    RecommendationEngine = None
    RECOMMENDATION_MODULES_AVAILABLE = False

# Variable globale pour le moteur de recommandation
recommender_engine: Optional[RecommendationEngine] = None

def optimize_dataframe_memory(df):
    """
    Optimise la mémoire utilisée par un DataFrame pandas.
    """
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def initialize_recommendation_engine() -> Optional[RecommendationEngine]:
    """
    Initialise le moteur de recommandation avec gestion d'erreurs robuste.
    """
    global recommender_engine
    
    if recommender_engine is not None:
        return recommender_engine
    
    if not RECOMMENDATION_MODULES_AVAILABLE:
        logger.error("Recommendation modules not available")
        return None
    
    try:
        data_path = os.path.join(parent_dir, 'processed_data')
        
        articles_metadata_path = os.path.join(data_path, 'articles_metadata.json')
        with open(articles_metadata_path, 'r') as f:
            articles_metadata_list = [json.loads(line) for line in f]
        articles_metadata = pd.DataFrame(articles_metadata_list)
        articles_metadata = optimize_dataframe_memory(articles_metadata)

        embeddings_path = os.path.join(data_path, 'embeddings_optimized.pkl')
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        data_summary_path = os.path.join(data_path, 'data_summary.json')
        with open(data_summary_path, 'r') as f:
            data_summary = json.load(f)
        
        user_interactions_path = os.path.join(data_path, 'user_interactions.json')
        with open(user_interactions_path, 'r') as f:
            user_interactions_list = [json.loads(line) for line in f]
        user_interactions = pd.DataFrame(user_interactions_list)
        user_interactions = optimize_dataframe_memory(user_interactions)

        if articles_metadata is None or embeddings is None or data_summary is None or user_interactions is None:
            raise ValueError("Failed to load one or more data components from local files.")

        recommender_engine = RecommendationEngine(
            articles_metadata=articles_metadata,
            user_interactions=user_interactions,
            embeddings=embeddings,
            data_summary=data_summary
        )
        
        # Libérer la mémoire
        del articles_metadata_list
        del user_interactions_list
        gc.collect()

        logger.info("RecommendationEngine initialized successfully")
        return recommender_engine
        
    except Exception as e:
        logger.error(f"Error initializing RecommendationEngine: {e}", exc_info=True)
        return None

def validate_request_body(req_body: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Valide et convertit les paramètres de la requête.
    """
    if not req_body:
        return None, None, "Request body is empty"
    
    user_id = req_body.get('user_id')
    n_recommendations = req_body.get('n_recommendations')
    
    if user_id is None:
        return None, None, "Missing 'user_id' in request body"
    
    if n_recommendations is None:
        return None, None, "Missing 'n_recommendations' in request body"
    
    try:
        user_id = int(user_id)
        n_recommendations = int(n_recommendations)
    except (ValueError, TypeError):
        return None, None, "'user_id' and 'n_recommendations' must be valid integers"
    
    if user_id <= 0:
        return None, None, "'user_id' must be a positive integer"
    
    if n_recommendations <= 0 or n_recommendations > 100:
        return None, None, "'n_recommendations' must be between 1 and 100"
    
    return user_id, n_recommendations, None

def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint pour obtenir des recommandations pour un utilisateur.
    """
    logger.info('Processing recommendation request')
    
    # Vérification des modules requis
    if not RECOMMENDATION_MODULES_AVAILABLE:
        return func.HttpResponse(
            json.dumps({
                "error": "Service unavailable", 
                "message": "Recommendation modules not available"
            }),
            mimetype="application/json",
            status_code=503
        )
    
    # Initialiser le moteur de recommandation
    engine = initialize_recommendation_engine()
    if engine is None:
        return func.HttpResponse(
            json.dumps({
                "error": "Service unavailable",
                "message": "Recommendation engine could not be initialized"
            }),
            mimetype="application/json",
            status_code=503
        )
    
    # Parser le JSON
    try:
        req_body = req.get_json()
    except Exception as e:
        logger.warning(f"Invalid JSON: {e}")
        return func.HttpResponse(
            json.dumps({
                "error": "Invalid JSON",
                "message": "Please provide valid JSON body"
            }),
            mimetype="application/json",
            status_code=400
        )
    
    # Valider les paramètres
    user_id, n_recommendations, error_message = validate_request_body(req_body)
    if error_message:
        return func.HttpResponse(
            json.dumps({
                "error": "Invalid request",
                "message": error_message
            }),
            mimetype="application/json",
            status_code=400
        )
    
    # Générer les recommandations
    try:
        recommendations = engine.get_recommendations(user_id, n_recommendations)
        
        return func.HttpResponse(
            json.dumps({
                "user_id": user_id,
                "n_recommendations": n_recommendations,
                "recommendations": recommendations,
                "status": "success"
            }),
            mimetype="application/json",
            status_code=200
        )
        
    except ValueError as ve:
        return func.HttpResponse(
            json.dumps({
                "error": "User not found",
                "message": str(ve),
                "user_id": user_id
            }),
            mimetype="application/json",
            status_code=404
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }),
            mimetype="application/json",
            status_code=500
        )
