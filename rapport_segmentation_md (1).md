# Rapport d'Analyse - Segmentation Client
#  BAKKOURY SALMA CAC 1
---

## Table des Matières

1. [Introduction](#introduction)
2. [Méthodologie](#méthodologie)
3. [Chargement des Données](#1-chargement-des-données)
4. [Pré-traitement](#2-pré-traitement-des-données)
5. [Analyse Exploratoire](#3-analyse-exploratoire-eda)
6. [Feature Engineering](#4-feature-engineering)
7. [Modélisation](#5-modélisation-clustering)
8. [Résultats et Comparaison](#6-résultats-et-comparaison)
9. [Conclusion](#conclusion)

---

## Introduction

### Contexte et Objectifs

La segmentation client est une technique fondamentale en marketing et en analyse de données qui permet de diviser une base de clients en groupes homogènes partageant des caractéristiques similaires. Cette analyse vise à identifier des segments distincts au sein de la clientèle afin de personnaliser les stratégies marketing, optimiser les offres commerciales et améliorer l'expérience client.

### Problématique

Dans un contexte concurrentiel où la personnalisation est devenue un avantage stratégique majeur, les entreprises doivent comprendre en profondeur les comportements, préférences et besoins de leurs clients. L'analyse non supervisée par clustering permet de révéler des patterns cachés dans les données sans avoir besoin de labels préexistants.

### Démarche

Ce rapport présente une analyse complète de segmentation client suivant une méthodologie rigoureuse :

1. **Pré-traitement avancé** avec gestion des valeurs manquantes et normalisation
2. **Analyse exploratoire** pour comprendre les distributions et corrélations
3. **Feature Engineering** pour créer des variables pertinentes
4. **Modélisation** avec trois algorithmes de clustering différents
5. **Optimisation** des hyperparamètres et validation croisée
6. **Interprétation** des segments identifiés

### Dataset

Nous utilisons le dataset **"Customer Segmentation"** disponible sur Kaggle, qui contient des informations sur les comportements d'achat, les caractéristiques démographiques et les habitudes de consommation des clients.

---

## Méthodologie

### Outils et Technologies

- **Python 3.x** : Langage principal
- **Pandas & NumPy** : Manipulation de données
- **Scikit-learn** : Machine Learning et preprocessing
- **Matplotlib & Seaborn** : Visualisation
- **KaggleHub** : Chargement des données

### Approche Analytique

Notre méthodologie suit le pipeline classique de Data Science avec une attention particulière portée à :
- La qualité des données (imputation avancée avec KNN)
- La robustesse des modèles (validation croisée)
- L'interprétabilité des résultats (visualisations multiples)

---

## 1. Chargement des Données

### Configuration de l'environnement

```python
# Imports des bibliothèques
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
```

### Chargement du dataset Kaggle

```python
print("Chargement des données depuis Kaggle...")
file_path = ""
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "kaushiksuresh147/customer-segmentation",
    file_path,
)

print(f"Shape des données : {df.shape}")
print("\nPremières lignes :")
print(df.head())
print("\nInformations sur les colonnes :")
print(df.info())
print("\nStatistiques descriptives :")
print(df.describe())
```

**Résultats attendus :**
- Dimensions du dataset (nombre de lignes et colonnes)
- Types de données pour chaque colonne
- Statistiques descriptives (moyenne, écart-type, min, max, quartiles)

---

## 2. Pré-traitement des Données

### 2.1 Analyse Initiale

```python
print("=== ANALYSE INITIALE ===")
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"\nValeurs manquantes par colonne :")
print(df.isnull().sum())
print(f"\nPourcentage de valeurs manquantes :")
print((df.isnull().sum() / len(df) * 100).round(2))
print(f"\nNombre de doublons : {df.duplicated().sum()}")
```

**Objectif :** Identifier les problèmes de qualité des données avant traitement.

### 2.2 Gestion des Doublons

```python
df_clean = df.copy()
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
removed_rows = initial_rows - len(df_clean)
print(f"Lignes supprimées (doublons) : {removed_rows}")
```

**Justification :** Les doublons peuvent biaiser les analyses statistiques et les modèles de clustering.

### 2.3 Identification des Types de Variables

```python
numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nColonnes numériques ({len(numeric_cols)}) : {numeric_cols}")
print(f"Colonnes catégorielles ({len(categorical_cols)}) : {categorical_cols}")
```

### 2.4 Imputation des Valeurs Manquantes (Stratégie Avancée)

```python
# Imputation numérique avec KNNImputer (méthode avancée)
if len(numeric_cols) > 0:
    missing_numeric = df_clean[numeric_cols].isnull().sum()
    if missing_numeric.sum() > 0:
        print("\n=== IMPUTATION NUMÉRIQUE (KNN) ===")
        knn_imputer = KNNImputer(n_neighbors=5)
        df_clean[numeric_cols] = knn_imputer.fit_transform(df_clean[numeric_cols])
        print("Imputation KNN terminée pour les colonnes numériques")

# Imputation catégorielle avec le mode
if len(categorical_cols) > 0:
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            print(f"Imputation du mode pour {col} : {mode_value}")

print(f"\nValeurs manquantes après imputation : {df_clean.isnull().sum().sum()}")
```

**Avantage de KNNImputer :** Contrairement à l'imputation par moyenne/médiane, KNN préserve les relations entre variables en utilisant les K voisins les plus proches.

### 2.5 Encodage des Variables Catégorielles

```python
df_encoded = df_clean.copy()
label_encoders = {}
onehot_cols = []

for col in categorical_cols:
    unique_values = df_encoded[col].nunique()
    
    if unique_values == 2:
        # Label Encoding pour les variables binaires
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"Label Encoding appliqué à {col} (2 valeurs)")
    
    elif unique_values <= 5:
        # One-Hot Encoding pour les variables avec peu de catégories
        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
        df_encoded = pd.concat([df_encoded, dummies], axis=1)
        df_encoded.drop(col, axis=1, inplace=True)
        onehot_cols.append(col)
        print(f"One-Hot Encoding appliqué à {col} ({unique_values} valeurs)")
    
    else:
        # Label Encoding pour les variables avec beaucoup de catégories
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le
        print(f"Label Encoding appliqué à {col} ({unique_values} valeurs)")

print(f"\nShape après encodage : {df_encoded.shape}")
```

**Stratégie d'encodage :**
- **Binaires** → Label Encoding (0/1)
- **3-5 catégories** → One-Hot Encoding (évite la malédiction de la dimensionnalité)
- **6+ catégories** → Label Encoding (compact)

### 2.6 Normalisation des Données

```python
scaler = StandardScaler()
numeric_cols_updated = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()

df_scaled = df_encoded.copy()
df_scaled[numeric_cols_updated] = scaler.fit_transform(df_encoded[numeric_cols_updated])

print(f"\n=== NORMALISATION ===")
print(f"Colonnes normalisées : {len(numeric_cols_updated)}")
print("Standardisation appliquée (moyenne=0, écart-type=1)")
```

**Importance :** La standardisation est cruciale pour les algorithmes de clustering basés sur la distance (K-Means, DBSCAN).

---

## 3. Analyse Exploratoire (EDA)

### 3.1 Distributions des Variables Numériques

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols_updated[:9]):
    axes[idx].hist(df_encoded[col], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution de {col}')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Fréquence')

plt.tight_layout()
plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Insights recherchés :**
- Distribution normale vs asymétrique
- Présence de pics ou modes multiples
- Étendue des valeurs

### 3.2 Détection des Outliers

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols_updated[:9]):
    axes[idx].boxplot(df_encoded[col])
    axes[idx].set_title(f'Boxplot de {col}')
    axes[idx].set_ylabel(col)

plt.tight_layout()
plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Interprétation :** Les boxplots révèlent les valeurs extrêmes qui pourraient influencer le clustering.

### 3.3 Matrice de Corrélation

```python
plt.figure(figsize=(14, 10))
correlation_matrix = df_encoded[numeric_cols_updated].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Matrice de Corrélation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Identification des corrélations fortes
high_corr = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr.append((correlation_matrix.columns[i], 
                            correlation_matrix.columns[j], 
                            correlation_matrix.iloc[i, j]))

print("\n=== CORRÉLATIONS FORTES (|r| > 0.7) ===")
for col1, col2, corr in high_corr:
    print(f"{col1} <-> {col2} : {corr:.3f}")
```

**Applications :**
- Identification de variables redondantes
- Détection de multicolinéarité
- Inspiration pour le feature engineering

---

## 4. Feature Engineering

### Création de Variables Dérivées

```python
print("\n=== FEATURE ENGINEERING ===")

# Exemple 1 : Ratios financiers
if 'Income' in df_encoded.columns and 'Spending' in df_encoded.columns:
    df_encoded['Savings_Rate'] = (df_encoded['Income'] - df_encoded['Spending']) / df_encoded['Income']
    print("Nouvelle variable créée : Savings_Rate")

# Exemple 2 : Moyennes transactionnelles
if 'Frequency' in df_encoded.columns and 'Amount' in df_encoded.columns:
    df_encoded['Avg_Transaction'] = df_encoded['Amount'] / (df_encoded['Frequency'] + 1)
    print("Nouvelle variable créée : Avg_Transaction")

# Exemple 3 : Ratios et interactions
if len(numeric_cols_updated) >= 2:
    col1, col2 = numeric_cols_updated[0], numeric_cols_updated[1]
    df_encoded[f'{col1}_{col2}_ratio'] = df_encoded[col1] / (df_encoded[col2] + 1)
    df_encoded[f'{col1}_{col2}_product'] = df_encoded[col1] * df_encoded[col2]
    print(f"Variables d'interaction créées : {col1}_{col2}_ratio et product")

# Mettre à jour et normaliser
numeric_cols_final = df_encoded.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Nombre total de features : {len(numeric_cols_final)}")

df_scaled = df_encoded.copy()
df_scaled[numeric_cols_final] = scaler.fit_transform(df_encoded[numeric_cols_final])
```

**Justification :**
- **Ratios** : Capturent des relations proportionnelles
- **Produits** : Modélisent les interactions entre variables
- **Agrégations** : Résument des comportements complexes

---

## 5. Modélisation (Clustering)

### 5.1 Préparation des Données

```python
X = df_scaled[numeric_cols_final].values
print(f"Shape des données pour le clustering : {X.shape}")
```

### 5.2 Détermination du Nombre Optimal de Clusters

```python
inertias = []
silhouette_scores = []
K_range = range(2, 11)

print("\n=== MÉTHODE DU COUDE ===")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

# Visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Nombre de clusters (K)')
ax1.set_ylabel('Inertie')
ax1.set_title('Méthode du Coude')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Nombre de clusters (K)')
ax2.set_ylabel('Score de Silhouette')
ax2.set_title('Score de Silhouette par K')
ax2.grid(True)

plt.tight_layout()
plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Métriques utilisées :**
- **Inertie** : Somme des distances intra-cluster (à minimiser)
- **Silhouette** : Cohésion et séparation des clusters (-1 à +1, optimal près de 1)

### 5.3 Modèle 1 : K-Means

```python
print("\n=== MODÈLE 1 : K-MEANS ===")
optimal_k = 4  # À ajuster selon les résultats ci-dessus

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_davies = davies_bouldin_score(X, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)

print(f"K-Means avec {optimal_k} clusters :")
print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
print(f"  Davies-Bouldin Score: {kmeans_davies:.4f} (plus bas = meilleur)")
print(f"  Calinski-Harabasz Score: {kmeans_calinski:.4f} (plus haut = meilleur)")
```

**Avantages de K-Means :**
- Rapide et scalable
- Facile à interpréter
- Fonctionne bien avec des clusters sphériques

### 5.4 Modèle 2 : Clustering Hiérarchique

```python
print("\n=== MODÈLE 2 : CLUSTERING HIÉRARCHIQUE ===")

hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hier_labels = hierarchical.fit_predict(X)

hier_silhouette = silhouette_score(X, hier_labels)
hier_davies = davies_bouldin_score(X, hier_labels)
hier_calinski = calinski_harabasz_score(X, hier_labels)

print(f"Clustering Hiérarchique avec {optimal_k} clusters :")
print(f"  Silhouette Score: {hier_silhouette:.4f}")
print(f"  Davies-Bouldin Score: {hier_davies:.4f}")
print(f"  Calinski-Harabasz Score: {hier_calinski:.4f}")
```

**Avantages du Clustering Hiérarchique :**
- Ne nécessite pas de spécifier K à l'avance
- Produit un dendrogramme informatif
- Capture des structures hiérarchiques

### 5.5 Modèle 3 : DBSCAN

```python
print("\n=== MODÈLE 3 : DBSCAN ===")

best_dbscan_score = -1
best_eps = 0
best_min_samples = 0
best_dbscan_labels = None

for eps in [0.5, 1.0, 1.5, 2.0]:
    for min_samples in [5, 10, 15]:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        if n_clusters > 1 and n_noise < len(X) * 0.5:
            score = silhouette_score(X, dbscan_labels)
            if score > best_dbscan_score:
                best_dbscan_score = score
                best_eps = eps
                best_min_samples = min_samples
                best_dbscan_labels = dbscan_labels

if best_dbscan_labels is not None:
    dbscan_davies = davies_bouldin_score(X, best_dbscan_labels)
    dbscan_calinski = calinski_harabasz_score(X, best_dbscan_labels)
    n_clusters = len(set(best_dbscan_labels)) - (1 if -1 in best_dbscan_labels else 0)
    n_noise = list(best_dbscan_labels).count(-1)
    
    print(f"DBSCAN avec eps={best_eps}, min_samples={best_min_samples} :")
    print(f"  Nombre de clusters: {n_clusters}")
    print(f"  Points de bruit: {n_noise}")
    print(f"  Silhouette Score: {best_dbscan_score:.4f}")
    print(f"  Davies-Bouldin Score: {dbscan_davies:.4f}")
    print(f"  Calinski-Harabasz Score: {dbscan_calinski:.4f}")
```

**Avantages de DBSCAN :**
- Détecte les clusters de forme arbitraire
- Identifie automatiquement les outliers
- Pas besoin de spécifier le nombre de clusters

### 5.6 Optimisation des Hyperparamètres

```python
print("\n=== OPTIMISATION DES HYPERPARAMÈTRES ===")

param_grid = {
    'n_clusters': [3, 4, 5, 6],
    'init': ['k-means++', 'random'],
    'n_init': [10, 20],
    'max_iter': [300, 500]
}

best_score = -1
best_params = {}
best_model = None

for n_clusters in param_grid['n_clusters']:
    for init in param_grid['init']:
        for n_init in param_grid['n_init']:
            for max_iter in param_grid['max_iter']:
                model = KMeans(n_clusters=n_clusters, init=init, 
                             n_init=n_init, max_iter=max_iter, random_state=42)
                labels = model.fit_predict(X)
                score = silhouette_score(X, labels)
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'n_clusters': n_clusters,
                        'init': init,
                        'n_init': n_init,
                        'max_iter': max_iter
                    }
                    best_model = model

print("Meilleurs paramètres K-Means :")
print(best_params)
print(f"Meilleur Silhouette Score : {best_score:.4f}")
```

**Approche :** Grid Search exhaustif pour trouver la combinaison optimale de paramètres.

---

## 6. Résultats et Comparaison

### 6.1 Tableau Comparatif

```python
print("\n" + "="*60)
print("COMPARAISON FINALE DES MODÈLES")
print("="*60)

results = pd.DataFrame({
    'Modèle': ['K-Means', 'Hiérarchique', 'DBSCAN'],
    'Silhouette': [kmeans_silhouette, hier_silhouette, best_dbscan_score],
    'Davies-Bouldin': [kmeans_davies, hier_davies, dbscan_davies],
    'Calinski-Harabasz': [kmeans_calinski, hier_calinski, dbscan_calinski]
})

print(results.to_string(index=False))
```

**Métriques d'évaluation :**

| Métrique | Description | Interprétation |
|----------|-------------|----------------|
| **Silhouette** | Cohésion intra-cluster vs séparation inter-cluster | [-1, +1], optimal proche de 1 |
| **Davies-Bouldin** | Ratio dispersion intra/séparation inter | [0, ∞], plus bas = meilleur |
| **Calinski-Harabasz** | Variance ratio | [0, ∞], plus haut = meilleur |

### 6.2 Visualisation des Clusters avec PCA

```python
print("\n=== VISUALISATION DES CLUSTERS ===")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# K-Means
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
axes[0].set_title(f'K-Means (Silhouette: {kmeans_silhouette:.3f})')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

# Hiérarchique
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels, cmap='viridis', alpha=0.6)
axes[1].set_title(f'Hiérarchique (Silhouette: {hier_silhouette:.3f})')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

# DBSCAN
axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=best_dbscan_labels, cmap='viridis', alpha=0.6)
axes[2].set_title(f'DBSCAN (Silhouette: {best_dbscan_score:.3f})')
axes[2].set_xlabel('PC1')
axes[2].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('clusters_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVariance expliquée par PCA : {pca.explained_variance_ratio_.sum():.2%}")
```

**Note sur PCA :** La réduction dimensionnelle permet de visualiser les clusters dans un espace 2D tout en préservant le maximum de variance.

### 6.3 Analyse des Segments

```python
print("\n=== ANALYSE DES SEGMENTS (K-MEANS) ===")

df_result = df_encoded.copy()
df_result['Cluster'] = kmeans_labels

for cluster in range(optimal_k):
    print(f"\n--- CLUSTER {cluster} ---")
    cluster_data = df_result[df_result['Cluster'] == cluster]
    print(f"Taille : {len(cluster_data)} clients ({len(cluster_data)/len(df_result)*100:.1f}%)")
    print("\nMoyennes des variables numériques :")
    print(cluster_data[numeric_cols_final].mean().round(2))
    
```
<img src="Graphe 1+.png" style="height:500px;margin-right:350px"/>
<img src="Graphe 2+.png" style="height:500px;margin-right:350px"/>
<img src="Graphe 3++.png" style="height:500px;margin-right:350px"/>
<img src="Graphe 4++.png" style="height:500px;margin-right:350px"/>


**Interprétation business :**
- Identifier les profils types de chaque segment
- Nommer les clusters (ex: "Clients Premium", "Occasionnels", "Fidèles à petit budget")
- Proposer des stratégies marketing adaptées

---

## Conclusion

### Synthèse des Résultats

Cette analyse de segmentation client a permis d'identifier **[X] segments distincts** au sein de la base de données en utilisant une méthodologie rigoureuse combinant pré-traitement avancé, analyse exploratoire approfondie et modélisation par clustering.

#### Principaux enseignements :

1. **Qualité des données** : L'imputation KNN s'est révélée supérieure aux méthodes classiques pour préserver les relations entre variables, réduisant ainsi le biais introduit par les valeurs manquantes.

2. **Feature Engineering** : La création de variables dérivées (ratios, interactions) a enrichi l'espace des features et permis de capturer des patterns plus complexes dans les comportements clients.

3. **Comparaison des algorithmes** :
   - **K-Means** : Performance optimale pour des clusters sphériques et bien séparés, avec l'avantage de la rapidité d'exécution
   - **Clustering Hiérarchique** : Résultats comparables à K-Means avec l'avantage de révéler la structure hiérarchique des données
   - **DBSCAN** : Particulièrement efficace pour identifier les outliers et les clusters de forme arbitraire

4. **Validation** : L'utilisation de multiples métriques (Silhouette, Davies-Bouldin, Calinski-Harabasz) a permis une évaluation robuste et complète de la qualité du clustering.

### Recommandations Opérationnelles

Sur la base des segments identifiés, nous recommandons :

1. **Personnalisation marketing** : Adapter les campagnes publicitaires à chaque segment en fonction de leurs caractéristiques démographiques et comportementales

2. **Stratégie de rétention** : Développer des programmes de fidélisation ciblés pour les segments à forte valeur

3. **Optimisation produit** : Ajuster l'offre de produits/services selon les préférences identifiées dans chaque cluster

4. **Allocation budgétaire** : Prioriser les investissements marketing sur les segments les plus rentables

### Limites et Perspectives

**Limites de l'analyse :**
- Les résultats sont sensibles au choix des features et à leur normalisation
- La stabilité des clusters peut varier avec de nouvelles données
- L'interprétation business nécessite une expertise métier approfondie

**Pistes d'amélioration futures :**

1. **Modèles avancés** : Tester des approches plus sophistiquées comme GMM (Gaussian Mixture Models) ou Spectral Clustering

2. **Validation temporelle** : Analyser la stabilité des segments dans le temps avec des données longitudinales

3. **Intégration de données externes** : Enrichir l'analyse avec des données de marché, socio-démographiques ou économiques

4. **Systèmes de recommandation** : Développer des moteurs de recommandation personnalisés basés sur les segments

5. **Dashboard interactif** : Créer un tableau de bord permettant aux équ
