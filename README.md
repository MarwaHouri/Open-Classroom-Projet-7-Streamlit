# Dashboard Interactif -Implémenter un modèle de scoring
## Description des fichiers
* **getData.py** : Code Python pour copier les fichiers source dans la racine du dossier
* **P7_Streamlit.py** : Code python pour l'implémentation du Dashboard sur Streamlit
* **P7_Flask.py** : Code python pour la définition des routes sur Flask RestFull API
* **requirements.txt** : Fichier txt contenant les librairies de dépendance nécessaires pour le projet
* **flask_routes.txt** : Fichier txt contenent la documentation des enpoints de l'API (ces informations sont égalements disponibles sur la page d'accueil de l'API)
* Les données du projet sont également disponibles sur le lien suivant https://drive.google.com/file/d/1dOWkgkL0rVAg8PJ8QE9oVkhU2Uy3rtGp/view?usp=share_link



## Commandes pour exécution 
* Prerequis : Python3 
* Pour installer les librairies de dépendance :
```
pip install -r requirements.txt
```
* Pour copier les fichier dans le dossier racine :
```
python getData.py
```
* Pour executer l’API :
```
nohup flask --app P7-Flask run --host=0.0.0.0 &
```
* Pour exécuter le Dashborad sur Streamlit :
```
nohup streamlit run P7-Streamlit.py &
```


