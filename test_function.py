#!/usr/bin/env python3
"""
Script de test pour l'Azure Function de recommandation
"""
import json
import sys
import os

# Ajouter le chemin pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'recommend'))

try:
    from recommend import main
    import azure.functions as func
    
    def test_function():
        """Test basique de l'Azure Function"""
        print("🧪 Test de l'Azure Function de recommandation...")
        
        # Créer une requête de test avec des paramètres GET
        test_req = func.HttpRequest(
            method='GET',
            url='http://localhost:7071/api/recommend?user_id=123&n_recommendations=5',
            headers={},
            params={'user_id': '123', 'n_recommendations': '5'},
            route_params={},
            body=b''
        )
        
        print("📤 Envoi de la requête de test...")
        print(f"   - user_id: 123")
        print(f"   - n_recommendations: 5")
        
        # Appeler la fonction
        try:
            response = main(test_req)
            print(f"📥 Réponse reçue:")
            print(f"   - Status Code: {response.status_code}")
            print(f"   - Content Type: {response.mimetype}")
            
            if response.status_code == 200:
                try:
                    response_data = json.loads(response.get_body().decode())
                    print(f"   - Recommandations: {len(response_data.get('recommendations', []))}")
                    print("✅ Test réussi !")
                except json.JSONDecodeError:
                    print("⚠️  Réponse non-JSON reçue")
            else:
                print(f"❌ Erreur: {response.get_body().decode()}")
                
        except Exception as e:
            print(f"❌ Erreur lors du test: {str(e)}")
            return False
        
        return True
    
    def test_invalid_params():
        """Test avec des paramètres invalides"""
        print("\n🧪 Test avec des paramètres invalides...")
        
        # Test sans user_id
        test_req = func.HttpRequest(
            method='GET',
            url='http://localhost:7071/api/recommend',
            headers={},
            params={},
            route_params={},
            body=b''
        )
        
        response = main(test_req)
        if response.status_code == 400:
            print("✅ Validation des paramètres fonctionne correctement")
        else:
            print(f"❌ Validation échouée: {response.status_code}")
    
    if __name__ == "__main__":
        print("=" * 50)
        print("🚀 TESTS DE L'AZURE FUNCTION")
        print("=" * 50)
        
        # Test principal
        success = test_function()
        
        # Test de validation
        test_invalid_params()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 Tests terminés avec succès !")
        else:
            print("❌ Certains tests ont échoué")
        print("=" * 50)

except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("Assurez-vous que tous les modules sont installés et que la structure est correcte.")
