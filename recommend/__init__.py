import logging
import json
import os
import sys
from typing import Optional, Dict, Any

import azure.functions as func

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import conditionnel des modules
try:
    from recommendation_engine.recommender import RecommendationEngine
    from recommendation_engine.data_loader import DataLoader
    RECOMMENDATION_MODULES_AVAILABLE = True
    logger.info("Recommendation modules imported successfully")
except ImportError as e:
    logger.error(f"Recommendation modules not available: {e}")
    RecommendationEngine = None
    DataLoader = None
    RECOMMENDATION_MODULES_AVAILABLE = False

# Import conditionnel d'Azure Storage
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_STORAGE_AVAILABLE = True
    logger.info("Azure Storage Blob module imported successfully")
except ImportError as e:
    logger.error(f"Azure Storage Blob module not available: {e}")
    BlobServiceClient = None
    AZURE_STORAGE_AVAILABLE = False

# Variable globale pour le moteur de recommandation
recommender_engine: Optional[RecommendationEngine] = None

def load_user_interactions_from_file() -> Optional[Dict[str, Any]]:
    """
    Charge les interactions utilisateur depuis un fichier local (fallback).
    """
    try:
        file_path = os.path.join(parent_dir, 'processed_data', 'user_interactions.json')
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                user_interactions = json.load(f)
            logger.info("User interactions loaded from local file")
            return user_interactions
        else:
            logger.warning(f"Local user interactions file not found: {file_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading user interactions from file: {e}")
        return {}

def load_user_interactions_from_blob() -> Optional[Dict[str, Any]]:
    """
    Charge les interactions utilisateur depuis Azure Blob Storage.
    """
    if not AZURE_STORAGE_AVAILABLE:
        logger.warning("Azure Storage not available, trying local file fallback")
        return load_user_interactions_from_file()
        
    try:
        connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connect_str:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set, using local file fallback")
            return load_user_interactions_from_file()
        
        container_name = "userinfosjson"
        blob_name = "user_interactions.json"
        
        logger.info(f"Loading user interactions from blob: {container_name}/{blob_name}")
        
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        blob_data = blob_client.download_blob().readall()
        user_interactions = json.loads(blob_data.decode('utf-8'))
        
        logger.info("User interactions loaded successfully from Blob Storage")
        return user_interactions
        
    except Exception as e:
        logger.warning(f"Error loading from blob, using local fallback: {e}")
        return load_user_interactions_from_file()

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
        logger.info(f"Loading local data from: {data_path}")
        
        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return None
        
        data_loader = DataLoader(data_path)
        articles_metadata = data_loader.load_articles_metadata()
        embeddings = data_loader.load_embeddings()
        data_summary = data_loader.load_data_summary()
        
        # Charger user_interactions (avec fallback)
        user_interactions = load_user_interactions_from_blob()
        if user_interactions is None:
            logger.error("Failed to load user interactions")
            return None
        
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

# Initialisation de l'application Azure Functions
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="recommend", methods=["POST"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
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
                "status": "success",
                "azure_storage_used": AZURE_STORAGE_AVAILABLE
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