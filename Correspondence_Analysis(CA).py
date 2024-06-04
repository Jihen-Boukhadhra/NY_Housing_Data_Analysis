######################  AFC  #######################

import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency
from google.colab import files
from scipy.stats import chi2_contingency

# Charger les données à partir du fichier Excel
df = pd.read_excel("NY-House-Dataset-Vars-Qualitatives.xlsx")

# Afficher les premières lignes du dataframe pour vérifier l'importation
print(df.head())

# Calcul de la table de contingence
contingency_table = pd.crosstab(df['TYPE'], df['SUBLOCALITY'])

print("\nTableau de contingence :")
print(contingency_table)
contingency_table.to_csv('contingency_table.csv', index=True)
files.download('contingency_table.csv')


# Calculer le tableau de contingence en fréquence
contingency_table_freq = contingency_table / contingency_table.values.sum()

print("\nTableau de contingence en fréquence :")
print(contingency_table_freq)
contingency_table_freq.to_csv('contingency_table_freq.csv', index=True)

files.download('contingency_table_freq.csv')

# Calculer les profils-lignes à partir du tableau de contingence
profils_lignes = contingency_table.div(contingency_table.sum(axis=1), axis=0)

print("\nTableau des profils_lignes:")
print(profils_lignes)
profils_lignes.to_csv('profils_lignes.csv', index=True)
files.download('profils_lignes.csv')

# Calculer les profils-colonnes à partir du tableau de contingence
profils_colonnes = contingency_table.div(contingency_table.sum(axis=0), axis=1)

print("\nTableau des profils_colonnes:")
print(profils_colonnes)
profils_colonnes.to_csv('profils_colonnes.csv', index=True)
files.download('profils_colonnes.csv')

# Effectuer le test du chi-deux
chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
print("\nTest du chi-deux :")
print("Statistique du test du chi-deux :", chi2_stat)
print("p-value :", p_value)
print("Degré de liberté :", dof)
print("Fréquences théoriques attendues :")
print(expected)


K = pd.read_csv("contingency_table.csv",header=0,index_col=0)
print(K)

# Calculer le nombre total d'observations
total_observations = np.sum(K.values)
khi2 = 21.03
# Effectuer le test du chi carré
chi2_stat, p_value, dof, expected = chi2_contingency(K)

# Afficher les résultats
print("Statistique du chi carré :", chi2_stat)
print("Valeur de p :", p_value)
if chi2_stat > khi2:
  print("On rejette H0 et on accepte H1, les deux vaiables sont dépendantes")
else:
  print("On accepte H0, les deux varaibles sont indépendantes")

# Calculer la matrice de fréquences
matrice_frequences = K / total_observations

# Afficher la matrice de fréquences
print("Matrice de fréquences F :")
print(matrice_frequences)


# Calculer les vecteurs de poids des lignes et des colonnes (distributions marginales)
vecteur_poids_lignes = np.sum(matrice_frequences, axis=1)
vecteur_poids_colonnes = np.sum(matrice_frequences, axis=0)

# Afficher les vecteurs de poids des lignes et des colonnes
print("Vecteur de poids des lignes (distribution marginale des lignes) :\n", vecteur_poids_lignes)
print("Vecteur de poids des colonnes (distribution marginale des colonnes) :\n", vecteur_poids_colonnes)

# Calculer la matrice des profils de lignes (distribution conditionnelle en ligne)
matrice_profil_lignes = K / K.values.sum(axis=1, keepdims=True)

# Afficher la matrice des profils de lignes
print("Matrice des profils de lignes (distribution conditionnelle en ligne) :")
print(matrice_profil_lignes)
profils_colonnes.to_csv('matrice_profils_lignes.csv', index=True)
files.download('matrice_profils_lignes.csv')

# Calculer la matrice des profils de colonnes (distribution conditionnelle en colonne)
matrice_profil_colonnes =  K / K.values.sum(axis=0, keepdims=True)

# Afficher la matrice des profils de colonnes
print("Matrice des profils de colonnes (distribution conditionnelle en colonne) :")
print(matrice_profil_colonnes)
profils_colonnes.to_csv('matrice_profils_colonnes.csv', index=True)
files.download('matrice_profils_colonnes.csv')

import matplotlib.pyplot as plt
# Définir les 11 modalités du type
Type= ['Coming Soon', 'Condo for Sale', 'Contingent', 'Co-Op for Sale', 'For Sale', 'ForeClosure', 'House for Sale', 'Land for Sale', 'Multi-family home for sale', 'Pending', 'Townhouse for sale']
# Définir les 11 sublocalities
Sublocality=['Bronx County', 'Brooklyn', 'East Bronx', 'Kings County', 'Manhattan', 'New York County', 'New York', 'Queens','Queens County', 'Richmond County', 'Staten Island', "The Bronx"]
# Créer une figure
fig, axs = plt.subplots(11, 1, figsize=(24, 30), sharex=True)
# Tracer chaque profil de ligne
for i in range(len(Type)):
    axs[i].bar(Type, np.array(matrice_profil_lignes)[i], color='skyblue')
    axs[i].set_ylabel('Fréquence')
    axs[i].set_title(f'Profil de ligne pour {Type[i]}')
# Ajouter une étiquette à l'axe des x au dernier sous-graphique
axs[-1].set_xlabel('Sublocality')
# Ajuster la mise en page
plt.tight_layout()
# Afficher le graphique
plt.show()

# Définir la matrice des profils de lignes (distribution conditionnelle en ligne)
matrice_profil_lignes_tab = np.array(matrice_profil_lignes)

# Calculer le profil de ligne moyen
profil_ligne_moyen = vecteur_poids_colonnes

# Afficher le profil de ligne moyen
print("Profil de ligne moyen :\n", profil_ligne_moyen)

# Calculer le profil de colonne moyen
profil_colonne_moyen = vecteur_poids_lignes

# Afficher le profil de ligne moyen
print("Profil de colonne moyen :\n", profil_colonne_moyen)

# Sélectionner deux modalités à comparer
modalite1 = matrice_profil_lignes_tab[0]  # 1ère modalité
modalite2 = matrice_profil_lignes_tab[3]  # 4ème modalité

modalite11 = matrice_profil_lignes_tab[1]  # 2ième modalité
modalite22 = matrice_profil_lignes_tab[7]  # 8ième modalité

# Calculer la distance de chi2 entre les deux modalités
chi2_distance = np.sum((modalite1 - modalite2)**2 / profil_ligne_moyen)
chi2_distance_1 = np.sum((modalite11 - modalite22)**2 / profil_ligne_moyen)
# Afficher la distance de chi2 entre les deux modalités
print("Distance de chi2 entre les deux modalités (Co-op for sale  et Coming Soon) :", chi2_distance)
print("Distance de chi2 entre les deux modalités (Coming Soon et Land For Sale) :", chi2_distance_1)
