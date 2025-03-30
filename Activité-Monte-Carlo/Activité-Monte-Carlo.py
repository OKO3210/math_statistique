import numpy as np
import matplotlib.pyplot as plt
import time

# Question 1: Simulation basique avec 10 000 points
np.random.seed(42)  # Pour reproductibilité

def estimate_pi(num_points):
    """Estime π par la méthode de Monte Carlo"""
    # Générer des points aléatoires dans le carré [-1,1] x [-1,1]
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    
    # Calculer les distances à l'origine
    distances = np.sqrt(x**2 + y**2)
    
    # Compter les points à l'intérieur du cercle (distance <= 1)
    points_in_circle = np.sum(distances <= 1)
    
    # Estimer π
    # Aire du cercle = π * r² = π * 1² = π
    # Aire du carré = 2²= 4
    # Ratio = π/4, donc π = 4 * ratio
    pi_estimate = 4 * points_in_circle / num_points
    
    return pi_estimate, x, y, distances

# Simulation avec 10 000 points
n_points = 10000
pi_estimate_10k, x_10k, y_10k, distances_10k = estimate_pi(n_points)

print(f"Estimation de π avec {n_points} points: {pi_estimate_10k}")
print(f"Valeur réelle de π: {np.pi}")
print(f"Erreur absolue: {abs(pi_estimate_10k - np.pi)}")
print(f"Erreur relative: {abs(pi_estimate_10k - np.pi) / np.pi * 100:.4f}%")

# Visualisation de la simulation
plt.figure(figsize=(10, 10))
plt.scatter(x_10k, y_10k, c=distances_10k <= 1, cmap='coolwarm', alpha=0.5, s=1)
circle = plt.Circle((0, 0), 1, fill=False, color='black')
plt.gca().add_patch(circle)
plt.axis('equal')
plt.title(f'Estimation de π par Monte Carlo avec {n_points} points\nπ ≈ {pi_estimate_10k:.6f}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Question 2: Impact du nombre de simulations
n_values = [100, 1000, 10000, 100000, 1000000]
pi_estimates = []
errors = []
times = []

for n in n_values:
    start_time = time.time()
    pi_est, _, _, _ = estimate_pi(n)
    end_time = time.time()
    
    pi_estimates.append(pi_est)
    errors.append(abs(pi_est - np.pi))
    times.append(end_time - start_time)
    
    print(f"n = {n:7d}, π ≈ {pi_est:.6f}, erreur = {abs(pi_est - np.pi):.6f}, temps = {end_time - start_time:.4f}s")

# Question 3: Erreur d'estimation
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.semilogx(n_values, pi_estimates, 'bo-')
plt.axhline(y=np.pi, color='r', linestyle='--', label='π réel')
plt.xlabel('Nombre de points')
plt.ylabel('Estimation de π')
plt.title('Convergence de l\'estimation de π')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.loglog(n_values, errors, 'ro-')
plt.xlabel('Nombre de points')
plt.ylabel('Erreur absolue')
plt.title('Erreur d\'estimation en fonction du nombre de points')
plt.grid(True)
plt.tight_layout()
plt.show()

# Question 5: Visualisation de la convergence
# Simulation en temps réel avec différents nombres de points
n_points_convergence = np.logspace(1, 6, 50).astype(int)
pi_estimates_convergence = []

for n in n_points_convergence:
    pi_est, _, _, _ = estimate_pi(n)
    pi_estimates_convergence.append(pi_est)

plt.figure(figsize=(10, 6))
plt.semilogx(n_points_convergence, pi_estimates_convergence, 'bo-', alpha=0.7)
plt.axhline(y=np.pi, color='r', linestyle='--', label='π réel')
plt.xlabel('Nombre de points (échelle log)')
plt.ylabel('Estimation de π')
plt.title('Convergence de l\'estimation de π par Monte Carlo')
plt.legend()
plt.grid(True)
plt.show()

# Question 6: Utilité et limites de Monte Carlo
print("\n===== UTILITÉ ET LIMITES DE MONTE CARLO =====")
print("Utilité de la méthode de Monte Carlo pour estimer π:")
print("\n1. Simplicité: L'approche est intuitive et facile à mettre en œuvre")
print("2. Parallélisable: On peut facilement distribuer les calculs sur plusieurs processeurs")
print("3. Applicable à des problèmes plus complexes: La méthode s'étend à des intégrales multidimensionnelles")

print("\nLimites:")
print("\n1. Convergence lente: L'erreur diminue en 1/√n, ce qui signifie qu'il faut multiplier le nombre de points par 100 pour gagner un chiffre de précision")
print("2. Précision limitée: Pour des applications nécessitant une grande précision, la méthode n'est pas adaptée")
print("3. Dépendance aux générateurs de nombres aléatoires: La qualité de l'estimation dépend de la qualité du générateur")

print("\nPour estimer π spécifiquement, il existe des méthodes bien plus efficaces, comme les séries de Bailey-Borwein-Plouffe ou la formule de Ramanujan, qui convergent beaucoup plus rapidement.")

print("\nLa méthode de Monte Carlo brille davantage pour des problèmes où:")
print("- Les méthodes déterministes sont difficiles à appliquer")
print("- On travaille dans des espaces à dimensions élevées")
print("- Une précision modérée est suffisante")