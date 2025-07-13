from azure.storage.blob import BlobServiceClient
from io import BytesIO

# Informations de connexion Azure
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "processed-data"

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
        return None
