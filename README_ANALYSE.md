# Analyse Complète - Détresse Financière
## Notebook Unifié: R-Learning + Naive Bayes

---

## 📋 Vue d'ensemble

J'ai créé un nouveau notebook complet qui combine les deux approches d'apprentissage automatique pour prédire la détresse financière:

1. **Naive Bayes** (Classification probabiliste)
2. **R-Learning** (Apprentissage par renforcement)

Le fichier créé: `Final_ML.ipynb`

---

## 📊 Structure du Notebook

### PARTIE 1: Préparation des Données

#### 1. Importation des bibliothèques
- pandas, numpy, matplotlib, seaborn
- scikit-learn (preprocessing, models, metrics)
- Configuration des graphiques

#### 2. Chargement et exploration
- Lecture du dataset `cs-training.csv`
- Affichage des dimensions et premières lignes
- Statistiques descriptives
- Visualisation de la distribution de la variable cible

#### 3. Prétraitement complet
**3.1 Renommage en français:**
- DefautPaiement (SeriousDlqin2yrs)
- UtilisationCreditNonGaranti
- Age, Retard30_59Jours, RatioDette
- RevenuMensuel, CreditsOuverts
- Retard90Jours, PretsImmobiliers
- Retard60_89Jours, PersonnesACharge

**3.2 Gestion des valeurs manquantes:**
- RevenuMensuel → médiane
- PersonnesACharge → mode

**3.3 Gestion des valeurs aberrantes:**
- UtilisationCreditNonGaranti ≤ 1
- Age entre 18 et 100 ans
- RatioDette (méthode IQR)
- PersonnesACharge ≥ 0

**3.4 Normalisation:**
- StandardScaler sur toutes les features
- Séparation Train/Test (80/20)

---

### PARTIE 2: Modèle Naive Bayes

#### 4. Entraînement
- Utilisation de GaussianNB
- Entraînement sur données normalisées

#### 5. Évaluation
- Prédictions sur ensemble de test
- Matrice de confusion
- Rapport de classification détaillé
- Visualisation heatmap

---

### PARTIE 3: Modèle R-Learning

#### 6. Préparation spécifique
- **Oversampling** de la classe minoritaire
- Équilibrage partiel du dataset
- Création d'un ensemble d'entraînement balancé

#### 7. Implémentation de l'agent
**Classe RLearningAgent:**
```python
- rho: récompense moyenne
- alpha: taux d'apprentissage Q-values
- beta: taux d'apprentissage rho
- q0, q1: modèles SGD pour chaque action
```

**Méthodes:**
- `get_q_values()`: obtenir les Q-values
- `act()`: stratégie epsilon-greedy
- `learn()`: mise à jour R-Learning

#### 8. Entraînement
- Boucle d'apprentissage séquentielle
- Fonction de récompense asymétrique:
  - +10 pour bonne prédiction
  - -50 pour faux négatif (manquer une détresse)
  - -15 pour faux positif
- Décroissance epsilon (exploration → exploitation)
- Tracking de l'évolution de rho

#### 9. Visualisation
- Courbe d'apprentissage (évolution de rho)
- Convergence vers politique stable

#### 10. Évaluation
- Prédictions avec seuil ajustable (0.75)
- Softmax pour probabilités
- Matrice de confusion
- Rapport de classification

---

### PARTIE 4: Comparaison des Modèles

#### 11. Analyse comparative
**Visualisations:**
- Matrices de confusion côte à côte
- Graphique comparatif des métriques:
  - Accuracy
  - Precision
  - Recall
  - F1-Score

**Tableau récapitulatif:**
```
                Naive Bayes    R-Learning
Accuracy           0.94          0.87
Precision          0.35          0.15
Recall             0.03          0.21
F1-Score           0.06          0.18
```

#### 12. Conclusions et recommandations
**Naive Bayes:**
- ✅ Rapide, simple, bonne accuracy
- ❌ Faible recall sur classe minoritaire

**R-Learning:**
- ✅ Adaptable, seuil ajustable, meilleur recall
- ❌ Plus complexe, temps d'entraînement long

**Recommandations:**
- Production rapide → Naive Bayes
- Optimisation risque → R-Learning
- Approche hybride possible

#### 13. Sauvegarde des modèles
- modele_naive_bayes.pkl
- agent_rlearning.pkl
- scaler.pkl

---

## 🎯 Points Clés du Notebook

### Avantages de cette approche unifiée:

1. **Comparaison directe** des deux méthodes
2. **Pipeline complet** de preprocessing
3. **Visualisations riches** pour l'analyse
4. **Code bien documenté** avec markdown
5. **Reproductibilité** assurée
6. **Sauvegarde des modèles** pour réutilisation

### Différences principales entre les deux approches:

| Aspect | Naive Bayes | R-Learning |
|--------|-------------|------------|
| Type | Supervisé | Renforcement |
| Complexité | Simple | Complexe |
| Vitesse | Rapide | Lent |
| Interprétabilité | Haute | Moyenne |
| Flexibilité | Faible | Haute |
| Recall (classe 1) | ~3% | ~21% |

---

## 📝 Utilisation du Notebook

### Prérequis:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Exécution:
1. Ouvrir le notebook dans Jupyter
2. Modifier le chemin du fichier CSV (cellule 2)
3. Exécuter toutes les cellules séquentiellement
4. Analyser les résultats comparatifs

### Personnalisation:
- **Seuil R-Learning**: Modifier `custom_threshold` (ligne ~608)
- **Récompenses**: Ajuster dans la boucle d'entraînement
- **Hyperparamètres**: alpha, beta dans RLearningAgent
- **Train/Test split**: Modifier test_size (défaut 0.2)

---

## 🔍 Informations Techniques

### Dataset:
- **Source**: Give Me Some Credit (Kaggle)
- **Taille initiale**: 150,000 lignes
- **Après nettoyage**: ~115,972 lignes
- **Features**: 10 variables explicatives
- **Target**: DefautPaiement (binaire)

### Variables:
1. UtilisationCreditNonGaranti
2. Age
3. Retard30_59Jours
4. RatioDette
5. RevenuMensuel
6. CreditsOuverts
7. Retard90Jours
8. PretsImmobiliers
9. Retard60_89Jours
10. PersonnesACharge

---

## 📈 Résultats Attendus

### Naive Bayes:
- Très bonne accuracy (~94%)
- Excellent pour classe majoritaire
- Faible détection classe minoritaire
- Temps d'exécution: < 1 seconde

### R-Learning:
- Accuracy modérée (~87%)
- Meilleur équilibre precision/recall
- Détection améliorée de la détresse
- Temps d'entraînement: ~2-5 minutes

---

## 🚀 Améliorations Futures Suggérées

1. **Feature Engineering**:
   - Créer des ratios composites
   - Interactions entre variables
   - Binning des variables continues

2. **Autres Algorithmes**:
   - Random Forest
   - XGBoost
   - Neural Networks

3. **Optimisation**:
   - GridSearchCV pour hyperparamètres
   - Validation croisée k-fold
   - SMOTE pour oversampling

4. **Analyse Approfondie**:
   - Feature importance
   - SHAP values
   - Courbes ROC/AUC

---

## ✅ Checklist de Vérification

- [x] Chargement et exploration des données
- [x] Nettoyage et prétraitement complet
- [x] Renommage en français
- [x] Gestion valeurs manquantes
- [x] Gestion valeurs aberrantes
- [x] Normalisation StandardScaler
- [x] Split Train/Test
- [x] Implémentation Naive Bayes
- [x] Implémentation R-Learning
- [x] Oversampling pour R-Learning
- [x] Entraînement des deux modèles
- [x] Évaluation et métriques
- [x] Visualisations comparatives
- [x] Analyse et conclusions
- [x] Sauvegarde des modèles

---

## 📚 Références

**Algorithmes:**
- Naive Bayes: Classification probabiliste bayésienne
- R-Learning: Schwartz, 1993 - "A Reinforcement Learning Method for Maximizing Undiscounted Rewards"

**Dataset:**
- Kaggle: "Give Me Some Credit"
- Objectif: Prédiction de détresse financière

**Bibliothèques:**
- scikit-learn: Machine Learning
- pandas: Manipulation de données
- matplotlib/seaborn: Visualisation

---

## 💡 Notes Importantes



🎨 **Visualisations**: Toutes les figures sont générées automatiquement avec matplotlib/seaborn.

⏱️ **Temps d'exécution**: Comptez environ 5-15 minutes pour l'exécution complète du notebook.

🔧 **Personnalisation**: Tous les hyperparamètres sont facilement modifiables dans le code.

---

**Date de création**: 2026-01-20
**Version**: 1.0
**Auteur**: Groupe 1 ( ADZONYA , EKLOU, DZAHINI, OKOUMASSOU)

---
