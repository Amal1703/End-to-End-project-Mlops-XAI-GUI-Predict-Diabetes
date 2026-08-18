
import streamlit as st
import joblib  # Utilisé pour charger le modèle sauvegardé
from flask import Flask, request, jsonify  # Flask est un micro-framework pour les applications web
import numpy as np
import pandas as pd
import torch
import requests
import torch
import json


URL_BASE = 'https://g1-833881892427.europe-west9.run.app'

def envoyer_pour_prediction(donnees):
    """ Envoie les données à l'API et récupère les prédictions. """
    response = requests.post(f"{URL_BASE}/predire", json=donnees)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def main():
    st.title("Application de prédiction de la diabéte")
    st.write("Veuillez charger un fichier excel contenant les données des pression.")

    # Téléchargement de fichier par l'utilisateur
    fichier = st.file_uploader("Choisissez un fichier excel", type='xlsx')
    if fichier is not None:
        # Chargement des données
        donnees = pd.read_excel(fichier)

        # Afficher les données chargées
        st.write("Données du sujet chargées :")
        st.write(donnees)

        if st.button("Prédire"):
            # Prédire chaque ligne des données chargées
            predictions = []
            P_value = donnees[['right pressure 1[N/cm²]', 'right pressure 2[N/cm²]', 
                                   'right pressure 3[N/cm²]','right pressure 4[N/cm²]', 
                                   'right pressure 5[N/cm²]', 'right pressure 6[N/cm²]',
                                   'right pressure 7[N/cm²]', 'right pressure 8[N/cm²]', 
                                   'right pressure 9[N/cm²]','right pressure 10[N/cm²]', 
                                   'right pressure 11[N/cm²]','right pressure 12[N/cm²]', 
                                   'right pressure 13[N/cm²]','right pressure 14[N/cm²]', 
                                   'right pressure 15[N/cm²]','right pressure 16[N/cm²]' ]] .iloc[2]
            
            #donnees_dict = {f"P{i+1}": P_value.iloc[i] for i in range(len(P_value))}
            donnees_dict = P_value.to_dict()
            print(donnees_dict)
            st.write(donnees_dict)    
            #donnees_api = P_value.to_dict()
            resultat = envoyer_pour_prediction(donnees_dict)
            
            if resultat:
               st.write("Prédictions :",resultat)
            else:
               st.write("Aucune prédiction reçue de l'API.")
            
               # predictions.append({'prediction': resultat['resultats']})
            # Affichage des résultats
            #st.write("Prédictions :",resultat)
            #st.table(resultat)
        else:
            st.write("Cliquez sur le bouton 'Prédire' pour obtenir les résultats.")

if __name__ == "__main__":
    
    main()
