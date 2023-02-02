# Find_job_Fengfeng_Asmae_David

[lien colab](https://colab.research.google.com/drive/1-rjmG619mRji9JbZW4cwM7GEeCMlJ005?usp=sharing)

## Choix effectués
- Imputation target : données insuffisantes donc imputation par la médianne
- certaines approximations dans le formatage des colonnes notamment pour "Intitulé du poste"
- Modèle retenue : Random forest regressor
- metrique d'erreur : r²_score, résultat 0.86 (salaire max) et 0.83 (salaire min)
- implémentation d'un pipeline pour l'entraînement du modèle de regression

## Description des fichiers : 
- clean_data.csv : dataset nettoyé utilisé pour l'entraînement du modèle de machine learning
- data.json : dataset brute
- Find_job_Cleaning : Nettoyage du dataset brute
- Find_job_Explo : Exploration des données
- Find_job_Modele : Entraînement du modèle définitif
- Find_job_Linear_regression : Premier modèle de regression linéaire entraîné, performances décevantes
- optimize_model.txt : contient les meilleurs résultats de l'optimization des paramètres de l'algo Random Forest Regressor  

## L'application web
- Conçu via Streamlit pour la rapidité de développement
- dossier web_app : contient les éléménts relatifs à l'application

### Précédure pour lancer l'application
```bash
cd web-app
```
```bash
streamlit run Find_job_app.py
```