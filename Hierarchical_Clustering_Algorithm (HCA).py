###############################  CAH  #############################
# Importer la librairie pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#scikit-learn
import sklearn
#classe StandardScaler pour standardisation (centrage et reduction)
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
# Charger les données à partir du fichier Excel sans en-têtes de colonnes
X = pd.read_excel("NY-House-Dataset-Num.xlsx")

# Afficher les données
print(X)

sc = StandardScaler()
#transformation – centrage-réduction
Z = sc.fit_transform(X)
print(Z)

#calculer la matrice de coorélation
Coor = X.corr()

#affichage de la matrice de corrélation en pourcentage
print(Coor)
eigenvalues, eigenvectors = np.linalg.eig(Coor)

diagonal_matrix = np.diag(eigenvalues)

#affichage des valeurs propres en ordre décroissant
print(diagonal_matrix)

# Calcul du pourcentage de variance expliquée par chacun des axes factoriels
tot = sum(eigenvalues)
var_exp = [(i / tot)*100 for i in sorted(eigenvalues, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print(pandas.DataFrame({'valprop':eigenvalues,'inertie':var_exp,'inertiecum':cum_var_exp}))


# Diagramme des valeurs propres
plt.figure(figsize=(10, 5))
plt.plot(eigenvalues, '-o')
plt.title('Diagramme des valeurs propres')
plt.xlabel('Indice')
plt.ylabel('Valeur propre')
plt.grid(True)
plt.show()

# Calcul des pourcentages cumulés
cumulative_percentages = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100

# Diagramme des pourcentages cumulés
plt.figure(figsize=(10, 5))
plt.plot(cumulative_percentages, '-o')
plt.title('Diagramme des pourcentages cumulés des valeurs propres')
plt.xlabel('Indice')
plt.ylabel('Pourcentage cumulé (%)')
plt.grid(True)
plt.show()

selected_eigenvectors = eigenvectors[:, :2]


Tp = np.dot(Z, selected_eigenvectors)


print("Coordonnées des individus dans le nouveau repère :")
print(Tp)



plt.figure(figsize=(8, 6))
plt.scatter(Tp[:, 0], Tp[:, 1], color='blue')


plt.xlabel('Axe factoriel 1')
plt.ylabel('Axe factoriel 2')
plt.title('Projection des individus sur les deux premiers axes factoriels')
plt.grid(True)
plt.show()


for i in range(len(X.columns)):
    plt.arrow(0, 0, eigenvectors[i, 0], eigenvectors[i, 1], color='green', width=0.000000000050, head_width=0.0000005)
    plt.text(eigenvectors[i, 0], eigenvectors[i, 1], X.columns[i], color='darkgreen', ha='right')

# Chargement du dataset
df = pd.read_excel("NY-House-Dataset-Num.xlsx")
# Prétraitement des données (normalisation, standardisation, etc.) si nécessaire
# Calcul de la matrice de liaison avec la méthode 'ward'
Z = linkage(df, method='ward')
# Affichage du dendrogramme pour aider à déterminer le nombre de classes
plt.figure(figsize=(20, 7))
plt.title("Dendrogramme")
dendrogram(Z, labels=df.index, leaf_rotation=90)
plt.xlabel('Observations')
plt.ylabel('Distance')
plt.show()
# Détermination du nombre de classes retenues  # Vous pouvez utiliser un critère comme le critère d'Elbow ou le critère de la distance maximale entre les classes # Ici, nous allons utiliser le critère de distance maximale
max_d = 1000  # Ajustez cette valeur selon votre dendrogramme
clusters = fcluster(Z, max_d, criterion='distance')
# Réaliser la partition en spécifiant le nombre de classes retenu
k = len(np.unique(clusters))  # Nombre de classes
print("Nombre de classes retenu:", k)
# Réduction de dimension pour la visualisation
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)
# Représenter graphiquement la partition obtenue sur les plans factoriels de projection (PCA)
plt.figure(figsize=(18, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Partition obtenue avec CAH (Nombre de classes = {})'.format(k))
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()
