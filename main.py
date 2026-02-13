import numpy as np
import matplotlib.pyplot as plt

# =============================
# INPUTS (interactive)
# =============================

print("=" * 50)
print("ZOMBIE OUTBREAK SIMULATION - Input Variables")
print("=" * 50)
print("Press ENTER to use default values\n")

# Time settings
print("\n--- TIME SETTINGS ---")
dt_input = input(f"Time step in days (default: 1.0): ").strip()
dt = float(dt_input) if dt_input else 1.0

t_max_input = input(f"Total simulation time in days (default: 1825): ").strip()
t_max = float(t_max_input) if t_max_input else 1825

# Initial populations
print("\n--- INITIAL POPULATIONS ---")
S0_input = input(f"Susceptible population (default: 2299000): ").strip()
S0 = float(S0_input) if S0_input else 2_299_000

E0_input = input(f"Exposed population (default: 500): ").strip()
E0 = float(E0_input) if E0_input else 500

Z0_input = input(f"Zombie population (default: 500): ").strip()
Z0 = float(Z0_input) if Z0_input else 500

V0_input = input(f"Vaccinated population (default: 0): ").strip()
V0 = float(V0_input) if V0_input else 0

F0_input = input(f"Food units (default: 2600000): ").strip()
F0 = float(F0_input) if F0_input else 2_600_000

# Human dynamics
print("\n--- HUMAN DYNAMICS ---")
r_input = input(f"Growth rate parameter (default: 0.00003): ").strip()
r = float(r_input) if r_input else 0.00003

nu_input = input(f"Vaccination rate (default: 0.0005): ").strip()
nu = float(nu_input) if nu_input else 0.0005

mu0_input = input(f"Base mortality rate (default: 0.02): ").strip()
mu0 = float(mu0_input) if mu0_input else 0.02

# Infection & zombies
print("\n--- INFECTION & ZOMBIES ---")
beta_input = input(f"Infection rate (default: 2.0e-8): ").strip()
beta = float(beta_input) if beta_input else 2.0e-8

sigma_input = input(f"Incubation rate (default: 0.25): ").strip()
sigma = float(sigma_input) if sigma_input else 0.25

delta_input = input(f"Zombie death rate (default: 0.02): ").strip()
delta = float(delta_input) if delta_input else 0.02

# Food & resources
print("\n--- FOOD & RESOURCES ---")
alpha_input = input(f"Food production rate (default: 2600000): ").strip()
alpha = float(alpha_input) if alpha_input else 2_600_000

beta_f_input = input(f"Food consumption rate (default: 1.0): ").strip()
beta_f = float(beta_f_input) if beta_f_input else 1.0

K0_input = input(f"Base carrying capacity (default: 2500000): ").strip()
K0 = float(K0_input) if K0_input else 2_500_000

Fc_input = input(f"Food at critical level (default: {F0}): ").strip()
Fc = float(Fc_input) if Fc_input else F0

print("\n" + "=" * 50)
print("Starting simulation...")
print("=" * 50)

# =============================
# SIMULATION (Euler's Method)
# =============================

time = np.arange(0, t_max + 1, dt)

S, E, Z, V, F = S0, E0, Z0, V0, F0

S_list, E_list, Z_list, V_list = [], [], [], []

for t in time:
    S_list.append(S)
    E_list.append(E)
    Z_list.append(Z)
    V_list.append(V)

    # Carrying capacity based on food
    K = K0 * (F / F0)

    # Starvation death rate
    mu = mu0 * max(0, 1 - F / Fc)

    # Rates of change
    dS = r * S * (1 - (S + E + V) / K) - beta * S * Z - nu * S - mu * S
    dE = beta * S * Z - sigma * E - mu * E
    dZ = sigma * E - delta * Z
    dV = nu * S - mu * V
    dF = alpha - beta_f * (S + E + V)

    # Euler update
    S += dS * dt
    E += dE * dt
    Z += dZ * dt
    V += dV * dt
    F += dF * dt

    # Prevent negative values
    S = max(S, 0)
    E = max(E, 0)
    Z = max(Z, 0)
    V = max(V, 0)
    F = max(F, 0)

# =============================
# GRAPH OUTPUT
# =============================

plt.figure(figsize=(12, 6))
plt.plot(time, S_list, label="Susceptible", linewidth=2)
plt.plot(time, E_list, label="Exposed", linewidth=2)
plt.plot(time, Z_list, label="Zombies", linewidth=2)
plt.plot(time, V_list, label="Vaccinated", linewidth=2)

plt.xlabel("Time (days)", fontsize=12)
plt.ylabel("Population", fontsize=12)
plt.title("Zombie Outbreak Simulation", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# Save the figure
output_file = "zombie_simulation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nGraph saved successfully as '{output_file}'")

# Display the figure
plt.show()