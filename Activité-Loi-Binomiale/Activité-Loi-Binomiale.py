import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Lecture du fichier
try:
    puces_df = pd.read_csv('Dataset_lot_puce_defectueuses.csv')
    
    # Aperçu des données
    print("Aperçu des données de lots de puces défectueuses:")
    print(puces_df.head())
    
    # Question 1: Lire le fichier
    
    # Préparation des données: compter le nombre de défectueux par lot
    # Convertir la colonne 'défectueux' en booléen (True/False) si ce n'est pas déjà le cas
    if puces_df['défectueux'].dtype == 'object':
        puces_df['est_defectueux'] = puces_df['défectueux'].apply(lambda x: x.lower() == 'true' or x.lower() == 'oui' or x == '1')
    else:
        puces_df['est_defectueux'] = puces_df['défectueux'] == 1
    
    # Grouper par lot_id et compter les puces défectueuses
    defectueux_par_lot = puces_df.groupby('lot_id')['est_defectueux'].sum().reset_index()
    defectueux_par_lot.columns = ['lot_id', 'nb_defectueux']
    
    # Question 2: Estimation globale - proportion moyenne de puces défectueuses
    total_puces = len(puces_df)
    total_lots = defectueux_par_lot['lot_id'].nunique()
    total_defectueux = puces_df['est_defectueux'].sum()
    proportion_defectueux = total_defectueux / total_puces
    
    print(f"\nEstimation globale:")
    print(f"Nombre total de puces: {total_puces}")
    print(f"Nombre total de lots: {total_lots}")
    print(f"Puces par lot: {total_puces / total_lots}")
    print(f"Nombre total de puces défectueuses: {total_defectueux}")
    print(f"Proportion moyenne de puces défectueuses: {proportion_defectueux:.4f} ({proportion_defectueux*100:.2f}%)")
    
    # Question 3: Variabilité - distribution du nombre de puces défectueuses par lot
    plt.figure(figsize=(10, 6))
    plt.hist(defectueux_par_lot['nb_defectueux'], bins=range(21), color='skyblue', edgecolor='black')
    plt.title('Distribution du nombre de puces défectueuses par lot')
    plt.xlabel('Nombre de puces défectueuses')
    plt.ylabel('Nombre de lots')
    plt.xticks(range(0, 21, 1))
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    
    # Question 4: Probabilité d'un lot problématique (>= 5 puces défectueuses)
    lots_problematiques = (defectueux_par_lot['nb_defectueux'] >= 5).sum()
    proba_lot_problematique = lots_problematiques / total_lots
    
    print(f"\nProbabilité d'un lot problématique (>= 5 puces défectueuses):")
    print(f"Nombre de lots problématiques: {lots_problematiques}")
    print(f"Probabilité empirique: {proba_lot_problematique:.4f} ({proba_lot_problematique*100:.2f}%)")
    
    # Comparaison avec la loi binomiale théorique (p=0.10, n=20)
    p_theorique = 0.10  # probabilité théorique de 10%
    n = 20  # 20 puces par lot
    proba_theorique = 1 - stats.binom.cdf(4, n, p_theorique)
    
    print(f"Probabilité théorique (loi binomiale avec p={p_theorique}): {proba_theorique:.4f} ({proba_theorique*100:.2f}%)")
    
    # Question 5: Décision qualité - pourcentage des lots rejetés
    pourcentage_rejete = proba_lot_problematique * 100
    
    print(f"\nDécision qualité:")
    print(f"Si l'entreprise rejette tout lot contenant 5 puces défectueuses ou plus,")
    print(f"{pourcentage_rejete:.2f}% des lots sera rejeté.")
    
    # Question 6: Amélioration - taux de défectuosité maximal pour un taux de rejet de 5%
    def calcul_taux_rejet(p_defectuosite, seuil_rejet=5):
        """Calcule le taux de rejet des lots étant donné p et le seuil de rejet"""
        # Probabilité que le nombre de défectueux soit >= seuil_rejet
        proba_rejet = 1 - stats.binom.cdf(seuil_rejet - 1, n, p_defectuosite)
        return proba_rejet
    
    # Tester plusieurs valeurs de p pour trouver celle qui donne un taux de rejet de 5%
    p_values = np.linspace(0.01, 0.20, 100)
    rejet_rates = [calcul_taux_rejet(p) * 100 for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, rejet_rates, 'b-')
    plt.axhline(y=5, color='r', linestyle='--', label='Objectif 5%')
    plt.xlabel('Taux de défectuosité p')
    plt.ylabel('Taux de rejet des lots (%)')
    plt.title('Taux de rejet en fonction du taux de défectuosité')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Trouver la valeur de p qui donne un taux de rejet proche de 5%
    # On peut faire une recherche par bisection pour plus de précision
    p_min, p_max = 0.01, 0.20
    target_rejet = 0.05  # 5%
    
    while p_max - p_min > 0.0001:
        p_mid = (p_min + p_max) / 2
        rejet_rate = calcul_taux_rejet(p_mid)
        
        if rejet_rate < target_rejet:
            p_min = p_mid
        else:
            p_max = p_mid
    
    p_optimal = (p_min + p_max) / 2
    rejet_rate_optimal = calcul_taux_rejet(p_optimal)
    
    print(f"\nAmélioration:")
    print(f"Pour un taux de rejet de 5%, le taux de défectuosité maximal d'une puce devrait être:")
    print(f"p = {p_optimal:.4f} ({p_optimal*100:.2f}%)")
    print(f"Ce qui donne un taux de rejet exactement de {rejet_rate_optimal*100:.2f}%")
    
    # Comparaison graphique entre distribution réelle et théorique
    counts = defectueux_par_lot['nb_defectueux'].value_counts().sort_index()
    x = np.arange(0, 21)
    
    # Distribution binomiale théorique avec p estimé empiriquement
    y_theorique_empirique = stats.binom.pmf(x, n, proportion_defectueux) * total_lots
    
    # Distribution binomiale théorique avec p=0.10 (valeur donnée dans l'énoncé)
    y_theorique_enonce = stats.binom.pmf(x, n, 0.10) * total_lots
    
    plt.figure(figsize=(12, 7))
    plt.bar(counts.index, counts.values, alpha=0.7, color='skyblue', label='Données réelles')
    plt.plot(x, y_theorique_empirique, 'ro-', label=f'Binomiale théorique (p={proportion_defectueux:.3f})')
    plt.plot(x, y_theorique_enonce, 'go-', label='Binomiale théorique (p=0.10)')
    plt.axvline(x=5, color='red', linestyle='--', label='Seuil de rejet ≥ 5')
    plt.title('Comparaison entre distribution réelle et théorique')
    plt.xlabel('Nombre de puces défectueuses par lot')
    plt.ylabel('Nombre de lots')
    plt.legend()
    plt.xticks(range(0, 21))
    plt.grid(axis='y', alpha=0.75)
    plt.show()

except FileNotFoundError:
    print("Le fichier Dataset_lot_puce_defectueuses.csv n'a pas été trouvé!")