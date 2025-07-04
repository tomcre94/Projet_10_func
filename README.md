# Projet_10_func

# Système de Recommandation d'Articles - My Content

## Description

Ce projet est développé dans le cadre d'une formation d'ingénieur en IA. Il vise à créer un MVP pour une application de recommandation d'articles et de livres pour encourager la lecture. Le projet est mené par le CTO et cofondateur de la startup My Content.

## Objectifs

- Développer un système de recommandation d'articles.
- Créer une application simple pour gérer et afficher les recommandations.
- Utiliser des architectures serverless pour le déploiement.
- Stocker et gérer le code via GitHub.

## Compétences Développées

- Manipulation de données et modélisation statistique.
- Déploiement d'architectures cloud.
- Gestion de produit et prise de décisions techniques.

## Architecture Technique

Le système utilise une architecture serverless avec Azure Functions pour exposer le système de recommandation. Deux options d'architecture sont envisagées :

1. Utilisation d'une API pour développer et exposer le système de recommandation.
2. Exploitation des fonctionnalités "Azure Blob Storage input binding" pour récupérer directement les fichiers et modèles.

## Données

Les données utilisées proviennent de Globo.com et contiennent des interactions utilisateurs avec des articles de news. Elles incluent :

- `clicks.zip` : Dossier avec des fichiers CSV contenant les sessions d'interaction des utilisateurs.
- `articles_metadata.csv` : Fichier CSV avec des métadonnées sur les articles publiés.
- `articles_embeddings.pickle` : Fichier Pickle contenant les embeddings des articles.

## Fonctionnalités

- Recommandation de cinq articles pour chaque utilisateur.
- Prise en compte de l'ajout de nouveaux utilisateurs et de nouveaux articles.

## Déploiement

### Prérequis

- Compte Azure avec accès à Azure Functions et Azure Blob Storage.
- Python 3.x et les bibliothèques nécessaires (Flask, Streamlit, etc.).
# Projet_10_func
