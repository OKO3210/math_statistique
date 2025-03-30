import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Paramètres du problème
lambda_normal = 30
capacite_employe = 5
lambda_samedi = lambda_normal * 1.2
lambda_pic = lambda_normal * 2

# Question 1: Dimensionnement des équipes
def proba_engorgement(nb_employes, lambda_val=lambda_normal):
    """
    Calcule la probabilité d'engorgement (plus de commandes que capacité)
    """
    capacite_totale = nb_employes * capacite_employe
    return 1 - stats.poisson.cdf(capacite_totale, lambda_val)

# Chercher le nombre d'employés nécessaire pour un risque < 10%
for nb_employes in range(1, 20):
    risque = proba_engorgement(nb_employes) * 100
    print(f"{nb_employes} employés: risque d'engorgement = {risque:.2f}%")
    if risque < 10:
        print(f"\nIl faut au minimum {nb_employes} employés pour maintenir le risque d'engorgement sous 10%.")
        break

# Question 2: Gestion des pics de commande
proba_pic = 1 - stats.poisson.cdf(40, lambda_normal)
print(f"\nProbabilité d'avoir plus de 40 commandes en une heure: {proba_pic:.4f} ({proba_pic*100:.2f}%)")

x = np.arange(0, 60)
y = stats.poisson.pmf(x, lambda_normal)

plt.figure(figsize=(10, 6))
plt.bar(x, y, alpha=0.7, color='skyblue')
plt.axvline(x=40, color='red', linestyle='--', label='Seuil: 40 commandes')
plt.title(f'Distribution de Poisson (λ={lambda_normal})')
plt.xlabel('Nombre de commandes par heure')
plt.ylabel('Probabilité')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Question 3: Délais de livraison
lambda_2h = lambda_normal * 2
capacite_camion = 60

proba_retard = 1 - stats.poisson.cdf(capacite_camion, lambda_2h)
print(f"\nProbabilité de dépasser la capacité du camion sur 2 heures: {proba_retard:.4f} ({proba_retard*100:.2f}%)")

# Question 4: Anticipation du week-end
lambda_samedi = lambda_normal * 1.2

for nb_employes in range(1, 20):
    risque_samedi = proba_engorgement(nb_employes, lambda_samedi) * 100
    print(f"{nb_employes} employés le samedi: risque d'engorgement = {risque_samedi:.2f}%")
    if risque_samedi < 5:
        print(f"\nPour le samedi, il faut au minimum {nb_employes} employés pour maintenir le risque d'engorgement sous 5%.")
        break

# Question 5: Planification d'urgence
lambda_pic = lambda_normal * 2

# Option 1: Augmenter le nombre d'employés
for nb_employes in range(1, 30):
    risque_pic = proba_engorgement(nb_employes, lambda_pic) * 100
    if risque_pic < 10:
        print(f"\nOption 1 - Black Friday: {nb_employes} employés nécessaires pour un risque < 10%")
        cout_option1 = nb_employes * 8 * 20
        print(f"Coût estimé: {cout_option1}€ pour la journée")
        break

# Option 2: Augmenter la cadence des camions (toutes les heures au lieu de 2h)
lambda_1h = lambda_pic
proba_retard_1h = 1 - stats.poisson.cdf(capacite_camion, lambda_1h)
print(f"\nOption 2 - Augmenter la cadence des camions: probabilité de retard = {proba_retard_1h*100:.2f}%")
cout_option2 = 4 * 150
print(f"Coût estimé: {cout_option2}€ pour la journée")

# Comparaison des distributions
plt.figure(figsize=(12, 7))
x = np.arange(0, 100)
y_normal = stats.poisson.pmf(x, lambda_normal)
y_samedi = stats.poisson.pmf(x, lambda_samedi)
y_pic = stats.poisson.pmf(x, lambda_pic)

plt.plot(x, y_normal, 'b-', label=f'Normal (λ={lambda_normal})')
plt.plot(x, y_samedi, 'g-', label=f'Samedi (λ={lambda_samedi})')
plt.plot(x, y_pic, 'r-', label=f'Black Friday (λ={lambda_pic})')
plt.axvline(x=capacite_employe * 7, color='black', linestyle='--', label='Capacité 7 employés')
plt.axvline(x=capacite_camion, color='purple', linestyle='--', label='Capacité camion (60)')

plt.title('Comparaison des distributions de Poisson selon les scénarios')
plt.xlabel('Nombre de commandes par heure')
plt.ylabel('Probabilité')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()