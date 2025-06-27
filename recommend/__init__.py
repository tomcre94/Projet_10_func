import logging
import json
import os
import sys

import logging
import json
import os
import sys

import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

# Ajouter le répertoire parent au PYTHONPATH pour importer recommendation_engine
# Cela suppose que recommendation_engine et processed_data sont au même niveau que le dossier 'recommend'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from recommendation_engine.recommender import RecommendationEngine
from recommendation_engine.data_loader import DataLoader

# Initialisation du moteur de recommandation en dehors de la fonction principale
# pour optimiser le cold start.
# Les chemins d'accès aux données doivent être relatifs au répertoire racine de l'application de fonction.
# Assurez-vous que 'processed_data' est copié à la racine de l'application de fonction.
recommender_engine = None
try:
    # Charger les données locales (articles_metadata, embeddings, data_summary)
    data_path = os.path.join(parent_dir, 'processed_data')
    logging.info(f"Attempting to load local data from: {data_path}")
    data_loader = DataLoader(data_path)
    articles_metadata = data_loader.load_articles_metadata()
    embeddings = data_loader.load_embeddings()
    data_summary = data_loader.load_data_summary()

    # Charger user_interactions depuis Azure Blob Storage
    logging.info("Attempting to load user_interactions from Azure Blob Storage.")
    connect_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name = "userinfosjson" # Nom du conteneur fourni par l'utilisateur
    blob_name = "user_interactions.json"

    if not connect_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set.")

    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Télécharger le contenu du blob
    blob_data = blob_client.download_blob().readall()
    user_interactions = json.loads(blob_data)
    logging.info("user_interactions loaded from Blob Storage successfully.")

    recommender_engine = RecommendationEngine(
        articles_metadata=articles_metadata,
        user_interactions=user_interactions,
        embeddings=embeddings,
        data_summary=data_summary
    )
    logging.info("RecommendationEngine initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing RecommendationEngine: {e}")
    recommender_engine = None # Set to None to indicate failure

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="recommend", methods=["post"])
def recommend(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    if recommender_engine is None:
        return func.HttpResponse(
            "Internal server error: Recommendation engine not initialized.",
            status_code=500
        )

    try:
        req_body = req.get_json()
    except ValueError:
        return func.HttpResponse(
             "Please pass a JSON body in the request.",
             status_code=400
        )

    user_id = req_body.get('user_id')
    n_recommendations = req_body.get('n_recommendations')

    if user_id is None or n_recommendations is None:
        return func.HttpResponse(
             "Please pass 'user_id' and 'n_recommendations' in the request body.",
             status_code=400
        )

    try:
        user_id = int(user_id)
        n_recommendations = int(n_recommendations)
    except ValueError:
        return func.HttpResponse(
             "Invalid input: 'user_id' and 'n_recommendations' must be integers.",
             status_code=400
        )

    try:
        recommendations = recommender_engine.get_recommendations(user_id, n_recommendations)
        return func.HttpResponse(
            json.dumps(recommendations),
            mimetype="application/json",
            status_code=200
        )
    except ValueError as ve:
        logging.warning(f"Recommendation error: {ve}")
        return func.HttpResponse(
            str(ve),
            status_code=404 # User not found or similar data-related error
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return func.HttpResponse(
            "An unexpected error occurred during recommendation processing.",
            status_code=500
        )
