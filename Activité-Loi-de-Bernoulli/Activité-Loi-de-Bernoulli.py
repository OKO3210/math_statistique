import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Lecture du fichier
try:
    composants_df = pd.read_csv('Dataset_composants_defectueux.csv')
    
    # Aperçu des données
    print("Aperçu des données de composants défectueux:")
    print(composants_df.head())
    
    # Vérification de la structure
    print("\nStructure des données:")
    print(composants_df.info())
    
    # Question 2: Estimer p la probabilité qu'un composant soit défectueux
    total_composants = len(composants_df)
    composants_defectueux = composants_df['defectueux'].sum()
    p_estime = composants_defectueux / total_composants
    
    print(f"\nEstimation de p (probabilité de défectuosité):")
    print(f"Total de composants: {total_composants}")
    print(f"Composants défectueux: {composants_defectueux}")
    print(f"p estimé: {p_estime:.4f} ({p_estime*100:.2f}%)")
    
    # Visualisation
    plt.figure(figsize=(10, 6))
    labels = ['Non défectueux', 'Défectueux']
    counts = [total_composants - composants_defectueux, composants_defectueux]
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'red'])
    plt.axis('equal')
    plt.title('Proportion de composants défectueux vs non défectueux')
    plt.show()
    
    # Intervalle de confiance pour p (approximation normale)
    z = 1.96
    marge_erreur = z * np.sqrt(p_estime * (1 - p_estime) / total_composants)
    ic_bas = max(0, p_estime - marge_erreur)
    ic_haut = min(1, p_estime + marge_erreur)
    
    print(f"\nIntervalle de confiance à 95% pour p: [{ic_bas:.4f}, {ic_haut:.4f}]")
    
    # Comparaison avec le taux attendu de 5%
    print(f"\nComparaison avec le taux attendu de 5%:")
    if 0.05 >= ic_bas and 0.05 <= ic_haut:
        print("Le taux de défectuosité attendu (5%) est dans l'intervalle de confiance.")
    else:
        print("Le taux de défectuosité attendu (5%) est en dehors de l'intervalle de confiance.")
        if p_estime > 0.05:
            print(f"Le taux de défectuosité est plus élevé que prévu (+{(p_estime-0.05)*100:.2f}%).")
        else:
            print(f"Le taux de défectuosité est plus bas que prévu (-{(0.05-p_estime)*100:.2f}%).")
    
except FileNotFoundError:
    print("Le fichier Dataset_composants_defectueux.csv n'a pas été trouvé!")