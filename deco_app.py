from qutip import *
import numpy as np
import matplotlib.pyplot as plt

# Parameters
energy_bias = 0.05       # Energy bias
tunnel_coupling = 0.2        # Tunneling coupling 
vibration_frequency = 0.03       # Active site vibration frequency
coupling_strength = 0.1        # Coupling strength 
num_vstates = 8      # Number of vibrational states 
decoherence_rate = 0.01   # Decoherence rate 

energies = np.linspace(0, 0.06, 7)
hydride_matrix = np.diag(energies)

for i in range(6):
    hydride_matrix[i][i+1] = tunnel_coupling
    hydride_matrix[i+1][i] = tunnel_coupling

hydride_ls = Qobj(hydride_matrix, dims=[[7], [7]])

annihilation_operator = destroy(num_vstates)  # Vibrational mode annihilation operator
vibration_h = vibration_frequency * annihilation_operator.dag() * annihilation_operator  # Harmonic oscillator Hamiltonian
population_site0 = basis(7,0) * basis(7,0).dag()
interaction_h = coupling_strength * tensor(population_site0, (annihilation_operator + annihilation_operator.dag())) # Interaction term between hydride and vibration

# Total system Hamiltonian 
H = tensor(hydride_ls, qeye(num_vstates)) + tensor(qeye(7), vibration_h) + interaction_h

# Initial state: hydride on donor site + vibrational ground state
initial_state = tensor(basis(7, 0), basis(num_vstates, 0))

# Observable: probability of finding hydride at NAD⁺ 
product_projector = tensor(basis(7, 6) * basis(7, 6).dag(), qeye(num_vstates))

# Time points for simulation
time_points = np.linspace(0, 200, 500)

decoherence_rates = np.linspace(0, 0.05, 6)

for decoherence_rate in decoherence_rates:

    # Collapse operator
    collapse_operator = [np.sqrt(decoherence_rate) * tensor(population_site0, qeye(num_vstates))]
    
    #solve the system
    result = mesolve(H, initial_state, time_points, collapse_operator, [product_projector])
    
    #plot and print
    plt.plot(time_points, result.expect[0], label= f"v = {decoherence_rate}")
    print(f"Final population at NAD+ for v = {decoherence_rate}: {result.expect[0][-1]:.3f}")



plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Vibrationally Assisted Hydride Tunneling in Alcohol Dehydrogenase")
plt.grid(True)
plt.legend()
plt.show()
