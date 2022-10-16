import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Dados de entrada - by Ansys ##################################
with open('01_temperatura_ansys_degrau_probe.txt') as f_input:
    data01 = [f.replace(",", ".") for f in f_input]
del data01[0]

T_prob_degrau = np.loadtxt(data01, usecols=2, unpack=True)
T_prob_degrau = np.insert(T_prob_degrau, 0, 20.)

with open('02_temperatura_ansys_1grau_probe.txt') as f_input:
    data02 = [f.replace(",", ".") for f in f_input]
del data02[0]

T_prob_1grau = np.loadtxt(data02, usecols=2, unpack=True)
T_prob_1grau = np.insert(T_prob_1grau, 0, 20.)

with open('03_temperatura_ansys_2grau_probe.txt') as f_input:
    data03 = [f.replace(",", ".") for f in f_input]
del data03[0]

T_prob_2grau = np.loadtxt(data03, usecols=2, unpack=True)
T_prob_2grau = np.insert(T_prob_2grau, 0, 20.)


# Dados de entrada - by simulação inversa ######################
with open('01_parametros_estimados_degrau.txt') as f_input:
    data11 = [f for f in f_input]
del data11[0]

t, Q_est_degrau, T_est_degrau = np.loadtxt(data11, usecols=(1, 2, 3), unpack=True)

with open('02_parametros_estimados_1grau.txt') as f_input:
    data12 = [f for f in f_input]
del data12[0]

Q_est_1grau, T_est_1grau = np.loadtxt(data12, usecols=(2, 3), unpack=True)

with open('03_parametros_estimados_2grau.txt') as f_input:
    data13 = [f for f in f_input]
del data13[0]

Q_est_2grau, T_est_2grau = np.loadtxt(data13, usecols=(2, 3), unpack=True)

# plot grafico da temperatura estimada e de prova
plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = '16'
plt.plot(t, T_est_degrau, 'bo', mfc='none', label='Temperatura estimada - Função Degrau')
plt.plot(t, T_prob_degrau, 'r', linewidth=2.5, label='Temperatura de prova - Função Degrau')
plt.plot(t, T_est_1grau, 'yo', mfc='none', label='Temperatura estimada - Função 1 grau')
plt.plot(t, T_prob_1grau, 'm', linewidth=2.5, label='Temperatura de prova - Função 1 grau')
plt.plot(t, T_est_2grau, 'co', mfc='none', label='Temperatura estimada - Função 2 grau')
plt.plot(t, T_prob_2grau, 'g', linewidth=2.5, label='Temperatura de prova - Função 2 grau')
plt.legend(fontsize=13)
plt.xlabel('Tempo (s)', fontsize=20)
plt.ylabel('Temperatura (°C)', fontsize=20)
plt.axis([t[0], t[-1], 0, 130])
plt.grid(True)
plt.text(7.9, 105, 'R² = %0.2f' % r2_score(T_prob_degrau, T_est_degrau))
plt.text(7.3, 85, 'R² = %0.2f' % r2_score(T_prob_1grau, T_est_1grau))
plt.text(5.8, 110, 'R² = %0.2f' % r2_score(T_prob_2grau, T_est_2grau))
plt.savefig('10_graf_temperaturas.png', dpi=150)
plt.show()



# plot grafico do delta temperatura estimada e de prova
plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = '16'
plt.plot(t, np.abs(T_est_degrau - T_prob_degrau), 'r', linewidth=2.5, label='Delta de Temperatura - Função Degrau')
plt.plot(t, np.abs(T_est_1grau - T_prob_1grau), 'm', linewidth=2.5, label='Delta de Temperatura - Função 1 grau')
plt.plot(t, np.abs(T_est_2grau - T_prob_2grau), 'g', linewidth=2.5, label='Delta de Temperatura - Função 2 grau')
plt.legend(fontsize=13)
plt.xlabel('Tempo (s)', fontsize=20)
plt.ylabel('Temperatura (°C)', fontsize=20)
plt.axis([t[0], t[-1], 0, 0.025])
plt.grid(True)
plt.savefig('10_graf_dif_temperaturas.png', dpi=150)
plt.show()

# plot grafico do fluxo de calor real e fluxo de calor estimado

def q_real_degrau(tm):
    return 5e5 * (tm >= 2.5) * (tm <= 7.5)

Q_prob_degrau = q_real_degrau(t) * 1e-3

def q_real_1grau(t):
    return (1e5 * t) * (t <= 5) + (1e6 - 1e5 * t) * (t > 5)

Q_prob_1grau = q_real_1grau(t) * 1e-3

def q_real_2grau(t):
    return -20000 * t**2 + 200000 * t

Q_prob_2grau = q_real_2grau(t) * 1e-3

plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = '16'
plt.plot(t, Q_est_degrau, 'bo', mfc='none', label='Fluxo de calor estimado - Função degrau')
plt.plot(t, Q_prob_degrau, 'r', linewidth=2.5, label='Fluxo de calor de prova - Função degrau')
plt.plot(t, Q_est_1grau, 'co', mfc='none', label='Fluxo de calor estimado - Função 1 grau')
plt.plot(t, Q_prob_1grau, 'g', linewidth=2.5, label='Fluxo de calor de prova - Função 1 grau')
plt.plot(t, Q_est_2grau, 'yo', mfc='none', label='Fluxo de calor estimado - Função 2 grau')
plt.plot(t, Q_prob_2grau, 'm', linewidth=2.5, label='Fluxo de calor de prova - Função 2 grau')
plt.legend(fontsize=13)
plt.xlabel('Tempo (s)', fontsize=20)
plt.ylabel('Fluxo de calor (kW/m²)', fontsize=20)
plt.axis([t[0], t[-1], -50, 550])
plt.grid(True)
plt.text(7.7, 450, 'R² = %0.2f' % r2_score(Q_prob_degrau, Q_est_degrau))
plt.text(8.2, 330, 'R² = %0.2f' % r2_score(Q_prob_1grau, Q_est_1grau))
plt.text(3, 270, 'R² = %0.2f' % r2_score(Q_prob_2grau, Q_est_2grau))
plt.savefig('10_graf_fluxos_de_calor.png', dpi=150)
plt.show()

# plot grafico do delta fluxo de calor estimada e de prova
plt.figure(figsize=(12, 8))
plt.rcParams['font.size'] = '16'
plt.plot(t, np.abs(Q_est_degrau - Q_prob_degrau), 'r', linewidth=2.5, label='Delta de Fluxo de Calor - Função Degrau')
plt.plot(t, np.abs(Q_est_1grau - Q_prob_1grau), 'm', linewidth=2.5, label='Delta de Fluxo de Calor - Função 1 grau')
plt.plot(t, np.abs(Q_est_2grau - Q_prob_2grau), 'g', linewidth=2.5, label='Delta de Fluxo de Calor - Função 2 grau')
plt.legend(fontsize=13)
plt.xlabel('Tempo (s)', fontsize=20)
plt.ylabel('Fluxo de calor (kW/m²)', fontsize=20)
plt.axis([t[0], t[-1], 0, 13])
plt.grid(True)
plt.savefig('10_graf_dif_flux_calor.png', dpi=150)
plt.show()



print('R² - fluxo de calor degrau:', r2_score(Q_prob_degrau, Q_est_degrau))
print('R² - fluxo de calor 1 grau:', r2_score(Q_prob_1grau, Q_est_1grau))
print('R² - fluxo de calor 2 grau:', r2_score(Q_prob_2grau, Q_est_2grau))

print('R² - temperatura fluxo de calor degrau:', r2_score(T_prob_degrau, T_est_degrau))
print('R² - temperatura fluxo de calor 1 grau:', r2_score(T_prob_1grau, T_est_1grau))
print('R² - temperatura fluxo de calor 2 grau:', r2_score(T_prob_2grau, T_est_2grau))


