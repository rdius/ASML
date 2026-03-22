# ASML
# Analyse Spatiale avec Machine Learning (ASML)
### Master — Institut International d'Ingénierie de l'Eau et de l'Environnement (2iE)

> **Responsable du cours** : Dr Rodrique KAFANDO  
> **Contact** : kafando.rodrique@gmail.com  
> **Niveau** : Master 1 · Semestre 1 · 6 ECTS  
> **Environnement** : 100 % Python — compatible Google Colab

---

## À propos du cours

Ce cours forme les étudiants à l'analyse spatiale automatisée appliquée aux enjeux de l'eau, de l'environnement et de la sécurité alimentaire en Afrique de l'Ouest. Il couvre l'ensemble de la chaîne : du chargement de données géospatiales brutes jusqu'à la production de cartes prédictives interprétables.

**Fil conducteur** : prédire les phases d'insécurité alimentaire (Cadre Harmonisé IPC) au Burkina Faso à partir de variables géospatiales ouvertes — GADM, Copernicus FAPAR, HOT OSM, WorldPop.

**Évaluation** : 40 % contrôle continu (quiz en ligne) · 60 % examen final écrit

---

## Structure du dépôt

```
ASML/
├── M1_Fondements_Python_SIG/
│   ├── M1_CM_Fondements_Python_SIG.docx      # Support de cours
│   ├── M1_CM_Notebook.ipynb                  # CM interactif
│   ├── M1_TP_Enonce_v2.ipynb                 # TP à compléter
│   ├── M1_TP_Correction_v2.ipynb             # Correction annotée
│   └── M1_Quiz_Evaluation.docx               # Quiz 30 pts
│
├── M2_Feature_Engineering_Spatial/
│   ├── M2_CM_Feature_Engineering_Spatial_v4.docx
│   ├── M2_CM_Notebook.ipynb
│   ├── M2_TP_Enonce_v4.ipynb
│   ├── M2_TP_Correction_v4.ipynb
│   └── M2_Quiz_Evaluation.docx
│
├── M3_ML_Classique_Spatial/
│   ├── M3_CM_ML_Classique_Spatial.docx
│   ├── M3_CM_Notebook.ipynb
│   ├── M3_TP_Enonce.ipynb
│   ├── M3_TP_Correction.ipynb
│   └── M3_Quiz_Evaluation.docx
│
├── M4_Deep_Learning_Spatial/                 # À venir
├── M5_LLM_VLM_GeoAI/                        # À venir
├── M6_Interpretabilite_Ethique/              # À venir
│
└── README.md
```

---

## Progression des modules

| # | Module | CM | TP | Quiz | Statut |
|---|--------|----|----|------|--------|
| **M1** | Fondements Python SIG | 4h | 6h | 30 pts | ✅ Complet |
| **M2** | Feature Engineering Spatial | 4h | 4h | 30 pts | ✅ Complet |
| **M3** | ML Classique Spatial | 4h | 6h | 30 pts | ✅ Complet |
| **M4** | Deep Learning Spatial + Earth Engine | 4h | 8h | — | 🔄 À venir |
| **M5** | LLM & VLM pour la GeoAI | 4h | 6h | — | 🔄 À venir |
| **M6** | Interprétabilité, Éthique et Limites | 4h | — | — | 🔄 À venir |

---

## Contenu par module

### M1 — Fondements Python SIG
Prise en main de l'écosystème Python géospatial. Chargement et manipulation de données vectorielles (GeoPandas), données raster (Rasterio), projections (CRS), visualisation statique et interactive.

**Objectifs clés** : GeoDataFrame · jointure spatiale · calcul NDVI · reprojection UTM · carte Folium

### M2 — Feature Engineering Spatial
Transformation de données géospatiales brutes en variables ML. Matrice de poids W, lag spatial, autocorrélation (Moran I / LISA), features de distance, statistiques zonales FAPAR, sélection par VIF.

**Objectifs clés** : matrice W Queen · lag spatial IPC · spatial block CV (justification) · FAPAR zonal · feature matrix

### M3 — ML Classique Spatial
Entraînement et évaluation de modèles sur la feature matrix M2. Random Forest, XGBoost, spatial cross-validation pour éviter le data leakage géographique, métriques adaptées (Kappa, F1-macro), interprétabilité (feature importance, SHAP), carte prédictive IPC.

**Objectifs clés** : RF + class_weight · spatial block CV · Cohen's Kappa · Moran I résidus · carte Folium IPC

---

## Données

Tous les notebooks intègrent un **mécanisme de fallback automatique** : ils tentent d'abord de charger les données réelles (GADM, HDX), puis basculent sur des données embarquées si la connexion échoue. Le cours fonctionne entièrement hors-ligne.

| Source | Contenu | Modules | URL |
|--------|---------|---------|-----|
| GADM 4.1 | Limites administratives BF (13 régions / 45 provinces) | M1, M2, M3 | [gadm.org](https://gadm.org) |
| Cadre Harmonisé (HDX) | Phases IPC insécurité alimentaire | M2, M3 | [data.humdata.org](https://data.humdata.org) |
| Copernicus FAPAR | Végétation biophysique 300m | M2 | [data.humdata.org](https://data.humdata.org) |
| HOT OSM Roads | Réseau routier BF | M2 | [data.humdata.org](https://data.humdata.org) |
| WorldPop / GRID3 | Population grillée 100m | M2 | [data.humdata.org](https://data.humdata.org) |

---

## Installation

```bash
# Option 1 — conda (recommandé)
conda create -n asml python=3.10
conda activate asml
conda install -c conda-forge geopandas rasterio folium contextily \
    libpysal esda rasterstats scikit-learn xgboost shap

# Option 2 — pip
pip install geopandas rasterio folium contextily \
    libpysal esda rasterstats scikit-learn xgboost shap
```

**Google Colab** : chaque notebook contient une cellule d'installation commentée à décommenter.

### Versions validées

| Bibliothèque | Version minimale | Rôle |
|-------------|-----------------|------|
| `geopandas` | ≥ 0.13 | Données vectorielles |
| `rasterio` | ≥ 1.3 | Données raster |
| `libpysal` | ≥ 4.7 | Matrice W, lag spatial |
| `esda` | ≥ 2.5 | Moran I, LISA |
| `rasterstats` | ≥ 0.19 | Statistiques zonales |
| `scikit-learn` | ≥ 1.3 | ML classique |
| `xgboost` | ≥ 1.7 | Gradient boosting |
| `shap` | ≥ 0.43 | Interprétabilité |
| `folium` | ≥ 0.14 | Cartes interactives |

---

## Comment utiliser les notebooks

Chaque module comprend trois types de notebooks avec des rôles distincts :

| Type | Fichier | Rôle | À exécuter seul ? |
|------|---------|------|------------------|
| **CM** | `M*_CM_Notebook.ipynb` | Version interactive du cours — lire et exécuter les exemples | Oui, en suivant le CM Word |
| **TP Énoncé** | `M*_TP_Enonce*.ipynb` | Exercices à compléter — **scénario différent du CM** | Oui, après avoir lu le CM |
| **TP Correction** | `M*_TP_Correction*.ipynb` | Correction annotée avec justifications | Après avoir tenté le TP |

> Les TP appliquent les **mêmes outils** que le CM sur des **problèmes différents** — ils évaluent la compréhension, pas la mémorisation.

---

## Fil conducteur technique

```
M1 → GeoDataFrame des provinces BF + raster NDVI
       ↓
M2 → M2_feature_matrix_BF.csv
     (45 provinces × 9 features géospatiales + ipc_phase)
       ↓
M3 → Modèle Random Forest entraîné
     + Carte IPC prédite (Folium HTML)
     + Baseline pour M4 Deep Learning
       ↓
M4 → CNN sur images Sentinel-2 + LSTM séries temporelles [à venir]
       ↓
M5 → Agents LLM/VLM pour la GeoAI [à venir]
       ↓
M6 → Interprétabilité SHAP + éthique de l'IA spatiale [à venir]
```

---

## Bibliographie sélective

- Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2).
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
- Chen, T. & Guestrin, C. (2016). XGBoost. *KDD 2016*.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications*, 11.
- Roberts, D. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8).
- Rey, S. & Anselin, L. (2010). PySAL: A Python Library of Spatial Analytical Methods. *RSSA*.
- Geographic Data Science with Python (open access) : [geographicdata.science](https://geographicdata.science)

---

## Licence

Les supports de cours sont mis à disposition à des fins pédagogiques. Toute réutilisation hors du cadre 2iE doit mentionner la source.

---

*Master ASML — Institut 2iE, Ouagadougou, Burkina Faso*
