import pandas as pd
import numpy as np
import os
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, processed_data_path="processed_data/"):
        self.processed_data_path = processed_data_path
        self.user_interactions_path = os.path.join(self.processed_data_path, 'user_interactions.json')
        self.articles_metadata_path = os.path.join(self.processed_data_path, 'articles_metadata.json')
        self.embeddings_optimized_path = os.path.join(self.processed_data_path, 'embeddings_optimized.pkl')
        self.data_summary_path = os.path.join(self.processed_data_path, 'data_summary.json')
        
        self.user_interactions = None
        self.articles_metadata = None
        self.embeddings_optimized = None
        self.data_summary = None

    def load_all_data(self):
        logging.info(f"Chargement des données depuis {self.processed_data_path}...")
        try:
            self.user_interactions = pd.read_json(self.user_interactions_path, lines=True)
            logging.info(f"Chargé user_interactions.json: {len(self.user_interactions)} interactions.")
            
            self.articles_metadata = pd.read_json(self.articles_metadata_path, lines=True)
            logging.info(f"Chargé articles_metadata.json: {len(self.articles_metadata)} articles.")
            
            with open(self.embeddings_optimized_path, 'rb') as f:
                self.embeddings_optimized = pickle.load(f)
            logging.info(f"Chargé embeddings_optimized.pkl: {self.embeddings_optimized.shape} dimensions.")
            
            with open(self.data_summary_path, 'r') as f:
                self.data_summary = json.load(f)
            logging.info("Chargé data_summary.json.")
            
            logging.info("Toutes les données préparées ont été chargées avec succès.")
            return True
        except Exception as e:
            logging.error(f"Erreur lors du chargement des données préparées: {e}")
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
