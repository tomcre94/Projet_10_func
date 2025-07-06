import pandas as pd
import numpy as np
import os
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, processed_data_path="processed_data/"):
        logger.info(f"Initializing DataLoader with data path: {processed_data_path}")
        self.processed_data_path = processed_data_path
        self.user_interactions_path = os.path.join(self.processed_data_path, 'user_interactions.json')
        self.articles_metadata_path = os.path.join(self.processed_data_path, 'articles_metadata.json')
        self.embeddings_optimized_path = os.path.join(self.processed_data_path, 'embeddings_optimized.pkl')
        self.data_summary_path = os.path.join(self.processed_data_path, 'data_summary.json')
        
        logger.info(f"Full path for user_interactions: {os.path.abspath(self.user_interactions_path)}")
        logger.info(f"Full path for articles_metadata: {os.path.abspath(self.articles_metadata_path)}")
        logger.info(f"Full path for embeddings_optimized: {os.path.abspath(self.embeddings_optimized_path)}")
        logger.info(f"Full path for data_summary: {os.path.abspath(self.data_summary_path)}")

        self.user_interactions = None
        self.articles_metadata = None
        self.embeddings_optimized = None
        self.data_summary = None

    def load_all_data(self):
        logger.info(f"Starting to load all data from {self.processed_data_path}...")
        try:
            logger.info(f"Loading user_interactions.json from {self.user_interactions_path}")
            self.user_interactions = pd.read_json(self.user_interactions_path, lines=True)
            logger.info(f"Loaded user_interactions.json: {len(self.user_interactions)} interactions, shape {self.user_interactions.shape}.")
            
            logger.info(f"Loading articles_metadata.json from {self.articles_metadata_path}")
            self.articles_metadata = pd.read_json(self.articles_metadata_path, lines=True)
            logger.info(f"Loaded articles_metadata.json: {len(self.articles_metadata)} articles, shape {self.articles_metadata.shape}.")
            
            logger.info(f"Loading embeddings_optimized.pkl from {self.embeddings_optimized_path}")
            with open(self.embeddings_optimized_path, 'rb') as f:
                self.embeddings_optimized = pickle.load(f)
            logger.info(f"Loaded embeddings_optimized.pkl: shape {self.embeddings_optimized.shape}, type {type(self.embeddings_optimized)}.")
            
            logger.info(f"Loading data_summary.json from {self.data_summary_path}")
            with open(self.data_summary_path, 'r') as f:
                self.data_summary = json.load(f)
            logger.info(f"Loaded data_summary.json: {self.data_summary}")
            
            logger.info("All data loaded successfully.")
            return True
        except FileNotFoundError as fnf_error:
            logger.error(f"File not found during data loading: {fnf_error}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"An error occurred during data loading: {e}", exc_info=True)
            return False

    def get_user_interactions(self):
        return self.user_interactions

    def get_articles_metadata(self):
        return self.articles_metadata

    def get_embeddings_optimized(self):
        return self.embeddings_optimized

    def get_data_summary(self):
        return self.data_summary

# Example usage (for testing purposes)
if __name__ == "__main__":
    loader = DataLoader()
    if loader.load_all_data():
        print("\nDonnées chargées avec succès. Aperçu:")
        print("User Interactions Head:")
        print(loader.get_user_interactions().head())
        print("\nArticles Metadata Head:")
        print(loader.get_articles_metadata().head())
        print("\nEmbeddings Optimized Shape:")
        print(loader.get_embeddings_optimized().shape)
        print("\nData Summary:")
        print(loader.get_data_summary())
