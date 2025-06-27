import logging
import json
import os
import sys
from typing import Optional, Dict, Any

import azure.functions as func
from azure.storage.blob import BlobServiceClient

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au PYTHONPATH pour importer recommendation_engine
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from recommendation_engine.recommender import RecommendationEngine
    from recommendation_engine.data_loader import DataLoader
except ImportError as e:
    logger.error(f"Failed to import recommendation modules: {e}")
    RecommendationEngine = None
    DataLoader = None

# Variable globale pour le moteur de recommandation
recommender_engine: Optional[RecommendationEngine] = None

def initialize_recommendation_engine() -> Optional[RecommendationEngine]:
    """
    Initialise le moteur de recommandation avec gestion d'erreurs robuste.
    """
    global recommender_engine
    
    if recommender_engine is not None:
        return recommender_engine
    
    if RecommendationEngine is None or DataLoader is None:
        logger.error("Recommendation modules not available")
        return None
    
    try:
        # Charger les données locales
        data_path = os.path.join(parent_dir, 'processed_data')
        logger.info(f"Loading local data from: {data_path}")
        
        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return None
        
        data_loader = DataLoader(data_path)
        articles_metadata = data_loader.load_articles_metadata()
        embeddings = data_loader.load_embeddings()
        data_summary = data_loader.load_data_summary()
        
        # Charger user_interactions depuis Azure Blob Storage
        user_interactions = load_user_interactions_from_blob()
        if user_interactions is None:
            logger.error("Failed to load user interactions from blob storage")
            return None
        
        # Initialiser le moteur de recommandation
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

def load_user_interactions_from_blob() -> Optional[Dict[str, Any]]:
    """
    Charge les interactions utilisateur depuis Azure Blob Storage.
    """
    try:
        connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connect_str:
            logger.error("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
            return None
        
        container_name = "userinfosjson"
        blob_name = "user_interactions.json"
        
        logger.info(f"Loading user interactions from blob: {container_name}/{blob_name}")
        
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        # Télécharger et parser le JSON
        blob_data = blob_client.download_blob().readall()
        user_interactions = json.loads(blob_data.decode('utf-8'))
        
        logger.info("User interactions loaded successfully from Blob Storage")
        return user_interactions
        
    except Exception as e:
        logger.error(f"Error loading user interactions from blob: {e}", exc_info=True)
        return None

def validate_request_body(req_body: Dict[str, Any]) -> tuple[Optional[int], Optional[int], Optional[str]]:
    """
    Valide et convertit les paramètres de la requête.
    
    Returns:
        tuple: (user_id, n_recommendations, error_message)
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

# Initialisation de l'application Azure Functions
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="recommend", methods=["POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    """
    Endpoint pour obtenir des recommandations pour un utilisateur.
    
    Expected JSON body:
    {
        "user_id": int,
        "n_recommendations": int
    }
    """
    logger.info('Processing recommendation request')
    
    # Initialiser le moteur de recommandation si nécessaire
    engine = initialize_recommendation_engine()
    if engine is None:
        logger.error("Recommendation engine not available")
        return func.HttpResponse(
            json.dumps({
                "error": "Recommendation service temporarily unavailable",
                "message": "The recommendation engine could not be initialized"
            }),
            mimetype="application/json",
            status_code=503
        )
    
    # Parser le JSON de la requête
    try:
        req_body = req.get_json()
    except Exception as e:
        logger.warning(f"Invalid JSON in request: {e}")
        return func.HttpResponse(
            json.dumps({
                "error": "Invalid JSON",
                "message": "Please provide a valid JSON body"
            }),
            mimetype="application/json",
            status_code=400
        )
    
    # Valider les paramètres
    user_id, n_recommendations, error_message = validate_request_body(req_body)
    if error_message:
        logger.warning(f"Request validation failed: {error_message}")
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
        logger.info(f"Generating {n_recommendations} recommendations for user {user_id}")
        recommendations = engine.get_recommendations(user_id, n_recommendations)
        
        response_data = {
            "user_id": user_id,
            "n_recommendations": n_recommendations,
            "recommendations": recommendations,
            "status": "success"
        }
        
        logger.info(f"Successfully generated recommendations for user {user_id}")
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200
        )
        
    except ValueError as ve:
        # Erreur métier (utilisateur non trouvé, etc.)
        logger.warning(f"Business logic error for user {user_id}: {ve}")
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
        # Erreur technique inattendue
        logger.error(f"Unexpected error processing recommendations for user {user_id}: {e}", exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "message": "An unexpected error occurred while processing your request"
            }),
            mimetype="application/json",
            status_code=500
        )