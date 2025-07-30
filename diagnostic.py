#!/usr/bin/env python3
"""
Script de diagnostic pour l'Azure Function
"""
import os
import sys

def check_azure_function_setup():
    """Vérifie la configuration de l'Azure Function"""
    print("🔍 DIAGNOSTIC DE L'AZURE FUNCTION")
    print("=" * 50)
    
    # Vérifier la structure des fichiers
    base_path = os.path.dirname(__file__)
    
    files_to_check = [
        ('recommend/__init__.py', 'Fichier principal de la fonction'),
        ('recommend/function.json', 'Configuration de la fonction'),
        ('host.json', 'Configuration de l\'hôte'),
        ('requirements.txt', 'Dépendances Python'),
        ('recommendation_engine/recommender.py', 'Moteur de recommandation'),
        ('config.py', 'Configuration'),
        ('../../processed_data/articles_metadata.json', 'Données articles'),
        ('../../processed_data/user_interactions.json', 'Données interactions'),
        ('../../processed_data/embeddings_optimized.pkl', 'Embeddings'),
        ('../../processed_data/data_summary.json', 'Résumé des données')
    ]
    
    print("📁 Vérification des fichiers requis:")
    all_files_exist = True
    
    for file_path, description in files_to_check:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            print(f"   ✅ {description}: {file_path}")
        else:
            print(f"   ❌ {description}: {file_path} (MANQUANT)")
            all_files_exist = False
    
    print("\n📦 Vérification des dépendances Python:")
    
    dependencies = [
        ('azure.functions', 'Azure Functions SDK'),
        ('azure.storage.blob', 'Azure Blob Storage'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('pickle', 'Pickle (built-in)'),
        ('json', 'JSON (built-in)')
    ]
    
    for module_name, description in dependencies:
        try:
            __import__(module_name)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} (MANQUANT)")
    
    print("\n🔧 Variables d'environnement:")
    azure_conn = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if azure_conn:
        print(f"   ✅ AZURE_STORAGE_CONNECTION_STRING: Configurée")
    else:
        print(f"   ⚠️  AZURE_STORAGE_CONNECTION_STRING: Non configurée (fallback aux fichiers locaux)")
    
    print("\n📋 Résumé:")
    if all_files_exist:
        print("   ✅ Tous les fichiers requis sont présents")
        print("   🚀 L'Azure Function devrait fonctionner en local")
        print("\n💡 Pour démarrer l'Azure Function en local:")
        print("   1. Installez Azure Functions Core Tools")
        print("   2. Exécutez: func start")
        print("   3. L'endpoint sera disponible sur: http://localhost:7071/api/recommend")
    else:
        print("   ❌ Certains fichiers requis sont manquants")
        print("   ⚠️  Veuillez corriger les problèmes avant de déployer")
    
    print("\n🌐 Test d'endpoint (exemple):")
    print("   GET http://localhost:7071/api/recommend?user_id=123&n_recommendations=5")
    
    print("=" * 50)

if __name__ == "__main__":
    check_azure_function_setup()
