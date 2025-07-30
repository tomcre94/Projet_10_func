import azure.functions as func
import logging
import json
import os
import pandas as pd
import numpy as np
import pickle
import gc
from typing import Optional, List, Dict
from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Variables globales
recommender_engine = None
logger = logging.getLogger(__name__)

# Flag pour vérifier la disponibilité des modules
RECOMMENDATION_MODULES_AVAILABLE = True

try:
    # Import des classes du moteur de recommandation
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from recommendation_engine.recommender import RecommendationEngine
    from config import RECOMMENDATION_CONFIG
    logger.info("Successfully imported RecommendationEngine and config")
except ImportError as e:
    logger.warning(f"Recommendation modules not available: {e}")
    RECOMMENDATION_MODULES_AVAILABLE = False
    # Configuration par défaut si l'import échoue
    RECOMMENDATION_CONFIG = {
        'weights': {
            'content_based': 0.4,
            'collaborative': 0.3, 
            'popularity': 0.3
        }
    }

# Informations de connexion Azure
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "processed-data"

def optimize_dataframe_memory(df):
    """Optimise la mémoire utilisée par un DataFrame pandas."""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

def download_blob_as_text(blob_service_client, blob_name):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    stream = blob_client.download_blob()
    return stream.readall().decode("utf-8")

def download_blob_as_bytes(blob_service_client, blob_name):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
    stream = blob_client.download_blob()
    return stream.readall()

def initialize_recommendation_engine() -> Optional[RecommendationEngine]:
    global recommender_engine
    
    if recommender_engine is not None:
        return recommender_engine
    
    if not RECOMMENDATION_MODULES_AVAILABLE:
        logger.error("Recommendation modules not available")
        return None

    try:
        # Vérifier si la chaîne de connexion Azure est disponible
        if not AZURE_STORAGE_CONNECTION_STRING:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set, trying local files...")
            return initialize_from_local_files()
        
        logger.info("Initializing from Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)

        # articles_metadata.json
        articles_metadata_text = download_blob_as_text(blob_service_client, "articles_metadata.json")
        articles_metadata_list = [json.loads(line) for line in articles_metadata_text.strip().split("\n")]
        articles_metadata = pd.DataFrame(articles_metadata_list)
        articles_metadata = optimize_dataframe_memory(articles_metadata)

        # embeddings_optimized.pkl
        embeddings_bytes = download_blob_as_bytes(blob_service_client, "embeddings_optimized.pkl")
        embeddings = pickle.loads(embeddings_bytes)

        # data_summary.json
        data_summary_text = download_blob_as_text(blob_service_client, "data_summary.json")
        data_summary = json.loads(data_summary_text)

        # user_interactions.json
        user_interactions_text = download_blob_as_text(blob_service_client, "user_interactions.json")
        user_interactions_list = [json.loads(line) for line in user_interactions_text.strip().split("\n")]
        user_interactions = pd.DataFrame(user_interactions_list)
        user_interactions = optimize_dataframe_memory(user_interactions)

        recommender_engine = RecommendationEngine(
            articles_metadata=articles_metadata,
            user_interactions=user_interactions,
            embeddings=embeddings,
            data_summary=data_summary
        )

        del articles_metadata_list, user_interactions_list
        gc.collect()

        logger.info("RecommendationEngine initialized successfully from Azure Blob Storage")
        return recommender_engine

    except Exception as e:
        logger.error(f"Error initializing RecommendationEngine from Azure: {e}", exc_info=True)
        logger.info("Falling back to local files...")
        return initialize_from_local_files()


def initialize_from_local_files() -> Optional[RecommendationEngine]:
    """Initialise le moteur de recommandation depuis les fichiers locaux"""
    global recommender_engine
    
    try:
        logger.info("Attempting to load from local processed_data folder...")
        
        # Chemins vers les fichiers locaux
        base_path = os.path.join(os.path.dirname(__file__), '..', '..', 'processed_data')
        
        # Articles metadata
        articles_metadata_path = os.path.join(base_path, 'articles_metadata.json')
        if not os.path.exists(articles_metadata_path):
            logger.error(f"Local file not found: {articles_metadata_path}")
            return None
        
        with open(articles_metadata_path, 'r', encoding='utf-8') as f:
            articles_metadata_list = [json.loads(line) for line in f.read().strip().split('\n')]
        articles_metadata = pd.DataFrame(articles_metadata_list)
        articles_metadata = optimize_dataframe_memory(articles_metadata)
        
        # Embeddings
        embeddings_path = os.path.join(base_path, 'embeddings_optimized.pkl')
        if not os.path.exists(embeddings_path):
            logger.error(f"Local file not found: {embeddings_path}")
            return None
        
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        
        # Data summary
        data_summary_path = os.path.join(base_path, 'data_summary.json')
        if not os.path.exists(data_summary_path):
            logger.error(f"Local file not found: {data_summary_path}")
            return None
        
        with open(data_summary_path, 'r', encoding='utf-8') as f:
            data_summary = json.load(f)
        
        # User interactions
        user_interactions_path = os.path.join(base_path, 'user_interactions.json')
        if not os.path.exists(user_interactions_path):
            logger.error(f"Local file not found: {user_interactions_path}")
            return None
        
        with open(user_interactions_path, 'r', encoding='utf-8') as f:
            user_interactions_list = [json.loads(line) for line in f.read().strip().split('\n')]
        user_interactions = pd.DataFrame(user_interactions_list)
        user_interactions = optimize_dataframe_memory(user_interactions)
        
        # Initialiser le moteur
        recommender_engine = RecommendationEngine(
            articles_metadata=articles_metadata,
            user_interactions=user_interactions,
            embeddings=embeddings,
            data_summary=data_summary
        )
        
        del articles_metadata_list, user_interactions_list
        gc.collect()
        
        logger.info("RecommendationEngine initialized successfully from local files")
        return recommender_engine
        
    except Exception as e:
        logger.error(f"Error initializing from local files: {e}", exc_info=True)
        return None


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Point d'entrée principal de l'Azure Function pour les recommandations.
    Utilise les paramètres GET et ne nécessite pas d'authentification.
    """
    try:
        logger.info('Azure Function started - Recommendation request received')
        
        # Récupérer les paramètres depuis la query string
        user_id = req.params.get('user_id')
        n_recommendations = req.params.get('n_recommendations', '5')
        
        # Validation des paramètres
        if not user_id:
            logger.warning('Missing user_id parameter')
            return func.HttpResponse(
                json.dumps({
                    "error": "user_id parameter is required",
                    "usage": "GET /api/recommend?user_id=123&n_recommendations=5"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        try:
            user_id = int(user_id)
            n_recommendations = int(n_recommendations)
            
            # Limiter le nombre de recommandations
            if n_recommendations < 1 or n_recommendations > 50:
                n_recommendations = 5
                
        except ValueError:
            logger.warning(f'Invalid parameter types: user_id={user_id}, n_recommendations={n_recommendations}')
            return func.HttpResponse(
                json.dumps({
                    "error": "user_id and n_recommendations must be valid integers"
                }),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f'Processing request for user_id={user_id}, n_recommendations={n_recommendations}')
        
        # Initialiser le moteur de recommandation
        recommender = initialize_recommendation_engine()
        if not recommender:
            logger.error('Failed to initialize recommendation engine')
            return func.HttpResponse(
                json.dumps({
                    "error": "Recommendation service temporarily unavailable",
                    "details": "Unable to initialize recommendation engine"
                }),
                status_code=503,
                mimetype="application/json"
            )
        
        # Générer les recommandations
        try:
            recommendations = recommender.recommend_articles(user_id, n_recommendations)
            
            if not recommendations:
                logger.info(f'No recommendations found for user {user_id}')
                return func.HttpResponse(
                    json.dumps({
                        "user_id": user_id,
                        "recommendations": [],
                        "count": 0,
                        "message": "No recommendations available for this user"
                    }),
                    status_code=200,
                    mimetype="application/json"
                )
            
            # Préparer la réponse
            response_data = {
                "user_id": user_id,
                "recommendations": recommendations,
                "count": len(recommendations),
                "message": "Recommendations generated successfully"
            }
            
            logger.info(f'Successfully generated {len(recommendations)} recommendations for user {user_id}')
            
            return func.HttpResponse(
                json.dumps(response_data, ensure_ascii=False),
                status_code=200,
                mimetype="application/json"
            )
            
        except Exception as rec_error:
            logger.error(f'Error generating recommendations: {str(rec_error)}', exc_info=True)
            return func.HttpResponse(
                json.dumps({
                    "error": "Failed to generate recommendations",
                    "details": str(rec_error)
                }),
                status_code=500,
                mimetype="application/json"
            )
    
    except Exception as e:
        logger.error(f'Unexpected error in main function: {str(e)}', exc_info=True)
        return func.HttpResponse(
            json.dumps({
                "error": "Internal server error",
                "details": str(e)
            }),
            status_code=500,
            mimetype="application/json"
        )
