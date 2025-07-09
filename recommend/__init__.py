import logging
import json
import os
import sys
import pickle
from typing import Optional, Dict, Any

import azure.functions as func
from azure.storage.blob import BlobServiceClient

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

def load_data_from_blob(connection_string, container_name, blob_name, is_pickle=False, is_json_lines=False):
    """
    Charge des données depuis un blob Azure.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_data = blob_client.download_blob().readall()
        if is_pickle:
            return pickle.loads(blob_data)
        elif is_json_lines:
            data = []
            for line in blob_data.decode('utf-8').splitlines():
                data.append(json.loads(line))
            return data
        else:
            return json.loads(blob_data.decode('utf-8'))
    except Exception as e:
        logger.error(f"Error loading {blob_name} from blob: {e}", exc_info=True)
        return None

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
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")

        articles_metadata = load_data_from_blob(connection_string, "processed-data", "articles_metadata.json", is_json_lines=True)
        embeddings = load_data_from_blob(connection_string, "processed-data", "embeddings_optimized.pkl", is_pickle=True)
        data_summary = load_data_from_blob(connection_string, "processed-data", "data_summary.json")
        user_interactions = load_data_from_blob(connection_string, "userinfosjson", "user_interactions.json", is_json_lines=True)

        if articles_metadata is None or embeddings is None or data_summary is None or user_interactions is None:
            raise ValueError("Failed to load one or more data components from Blob Storage.")

        recommender_engine = RecommendationEngine(
            articles_metadata=articles_metadata,
            user_interactions=user_interactions,
            embeddings=embeddings,
            data_summary=data_summary
        )
        
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
