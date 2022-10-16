import numpy as np
import scipy.sparse.linalg as spl
from scipy import sparse
from scipy.optimize import root_scalar
import time
import pandas as pd

t0 = time.time()

# Dados de entrada - by simulação Ansys ########################
with open('01_temperatura_ansys_degrau.txt') as f_input:
    data1 = [f.replace(",", ".") for f in f_input]
del data1[0]

tempos, T_input = np.loadtxt(data1, usecols=(1, 2), unpack=True)

tempos = np.insert(tempos, 0, 0.)
T_input = np.insert(T_input, 0, 20.)
del_t = tempos[1:] - tempos[0:-1]
passos = np.size(tempos)


# Propriedades #################################################
k_sol = 60.5                    # W/m.K
rho = 7850                      # kg/m**3
cp = 434                        # J/kg.K
alpha = k_sol / (rho * cp)      # m**2/s
# q_ger = 0                     # W/m**3
L_x = 100e-3                    # m
L_y = 9.5e-3                    # m
L_z = 60e-3                     # m
a_x = 50e-3                     # m  (fonte de calor)
a_z = 50e-3                     # m  (fonte de calor)


# Definição da malha ###########################################
n_x = 51   # nós
n_y = 21   # nós
n_z = 31   # nós
del_x = L_x / (n_x - 1)
del_y = L_y / (n_y - 1)
del_z = L_z / (n_z - 1)
n_ax = round(a_x/del_x) + 1
n_az = round(a_z/del_z) + 1


# Numero dos nós ###############################################
nos = np.arange(n_x * n_y * n_z).reshape([n_x, n_y, n_z])
# Nós centrais
nos_centrais = nos[1:n_x-1, 1:n_y-1, 1:n_z-1]
# Nós nas faces
nos_w = nos[0, :, :]
nos_e = nos[n_x - 1, :, :]
nos_n = nos[:, 0, :]
nos_s = nos[:, n_y - 1, :]
nos_f = nos[:, :, 0]
nos_b = nos[:, :, n_z - 1]
# Nós vizinhos as faces
nos_iw = nos[1, :, :]
nos_ie = nos[n_x - 2, :, :]
nos_in = nos[:, 1, :]
nos_is = nos[:, n_y - 2, :]
nos_if = nos[:, :, 1]
nos_ib = nos[:, :, n_z - 2]
# Nós nas arestas
nos_wf = np.intersect1d(nos_w, nos_f)
nos_wb = np.intersect1d(nos_w, nos_b)
nos_wn = np.intersect1d(nos_w, nos_n)
nos_ws = np.intersect1d(nos_w, nos_s)
nos_ef = np.intersect1d(nos_e, nos_f)
nos_eb = np.intersect1d(nos_e, nos_b)
nos_en = np.intersect1d(nos_e, nos_n)
nos_es = np.intersect1d(nos_e, nos_s)
nos_nf = np.intersect1d(nos_n, nos_f)
nos_nb = np.intersect1d(nos_n, nos_b)
nos_sf = np.intersect1d(nos_s, nos_f)
nos_sb = np.intersect1d(nos_s, nos_b)
# Nós nos vértices
no_wnf = np.intersect1d(nos_wn, nos_wf)
no_wnb = np.intersect1d(nos_wn, nos_wb)
no_wsf = np.intersect1d(nos_ws, nos_wf)
no_wsb = np.intersect1d(nos_ws, nos_wb)
no_enf = np.intersect1d(nos_en, nos_ef)
no_enb = np.intersect1d(nos_en, nos_eb)
no_esf = np.intersect1d(nos_es, nos_ef)
no_esb = np.intersect1d(nos_es, nos_eb)
# Nós na fonte de calor (face N)
nos_a = nos_n[0:n_ax, 0:n_az]
nos_ae = nos_a[n_ax - 1, :]
nos_ab = nos_a[:, n_az - 1]
no_aeb = nos_a[n_ax - 1, n_az - 1]


# Parametros Auxiliares ########################################
# considerou-se del_x, del_y e del_z constantes em suas respectivas direções
A_x = 1 / del_x**2
A_y = 1 / del_y**2
A_z = 1 / del_z**2
A_p = 2 * (A_x + A_y + A_z)

# Fatores de correção para os nós que estão na borda interna da fonte de calor
f_qe = 0.5 + a_x/del_x - round(a_x/del_x)
f_qb = 0.5 + a_z/del_z - round(a_z/del_z)


# Matriz A, B e T ##############################################
# Construção das diagonais da matriz A
A_arr_p = np.ones(n_x * n_y * n_z)*A_p
A_arr_e = np.ones(n_x * n_y * n_z)*(-A_x)
A_arr_w = np.ones(n_x * n_y * n_z)*(-A_x)
A_arr_s = np.ones(n_x * n_y * n_z)*(-A_y)
A_arr_n = np.ones(n_x * n_y * n_z)*(-A_y)
A_arr_b = np.ones(n_x * n_y * n_z)*(-A_z)
A_arr_f = np.ones(n_x * n_y * n_z)*(-A_z)

for i in nos_e.flatten():
    A_arr_w[i] = 0
for i in nos_ie.flatten():
    A_arr_w[i] = - 2 * A_x
for i in nos_w.flatten():
    A_arr_e[i] = 0
for i in nos_iw.flatten():
    A_arr_e[i] = - 2 * A_x
for i in nos_s.flatten():
    A_arr_n[i] = 0
for i in nos_is.flatten():
    A_arr_n[i] = - 2 * A_y
for i in nos_n.flatten():
    A_arr_s[i] = 0
for i in nos_in.flatten():
    A_arr_s[i] = - 2 * A_y
for i in nos_b.flatten():
    A_arr_f[i] = 0
for i in nos_ib.flatten():
    A_arr_f[i] = - 2 * A_z
for i in nos_f.flatten():
    A_arr_b[i] = 0
for i in nos_if.flatten():
    A_arr_b[i] = - 2 * A_z

# Matriz A esparsa
diagonais_A = [A_arr_p, A_arr_e, A_arr_w, A_arr_s, A_arr_n, A_arr_b, A_arr_f]
offset_A = [0, n_y * n_z, -n_y * n_z, n_z, -n_z, 1, -1]
A = sparse.dia_matrix((diagonais_A, offset_A), shape=(n_x * n_y * n_z, n_x * n_y * n_z))

# Matriz T
T_solved = np.zeros(passos * n_x * n_y * n_z).reshape([passos, n_x, n_y, n_z])
T_p0 = np.ones(n_x * n_y * n_z) * T_input[0]
T_p = T_p0
T_solved[0] = T_p0.reshape(n_x, n_y, n_z)


def calc_temp(q_fonte, temp_p, temp_prova, d_t, nx=10, ny=5, nz=10):
    global T_p
    a_t = (1 / (alpha * d_t))
    a_pt = np.ones(n_x * n_y * n_z) * a_t
    a = A + sparse.dia_matrix((a_pt, [0]), shape=(n_x * n_y * n_z, n_x * n_y * n_z))  # matriz A completa
    b_pq = 2 * q_fonte / (k_sol * del_y)
    temp_p0 = temp_p
    b = a_t * temp_p0
    for j in nos_a.flatten():
        if j == no_aeb:
            b[j] += b_pq * f_qe * f_qb
        elif j in nos_ae:
            b[j] += b_pq * f_qe
        elif j in nos_ab:
            b[j] += b_pq * f_qb
        else:
            b[j] += b_pq
    # Calculo da matriz de temperatura #
    temp_p, t_aux = spl.cgs(a, b, tol=1.0e-6)
    T_p = temp_p.reshape(n_x, n_y, n_z)
    return (T_p[nx, ny, nz] - temp_prova)*1e5


# Fluxo de calor estimado por passo ####################################################################################
q_est_0 = np.zeros(passos)
t_est = np.zeros(passos)

for i in np.arange(1, passos):
    res_q = root_scalar(calc_temp, args=(T_solved[i-1].flatten(), T_input[i], del_t[i-1]), bracket=[-1e5, 1e10],
                        method='brentq')
    q_est_0[i] = res_q.root
    t_est[i] = (tempos[i] + tempos[i - 1]) / 2
    T_solved[i] = T_p

T_probe = T_solved[:, 0, 5, 10]

df = pd.DataFrame({"Tempos (s)": tempos, "Fluxo de Calor (kW/m²)": q_est_0*1e-3,
                   "Temperatura de Prova (°C)": T_probe})
df.to_csv("01_parametros_estimados_degrau.txt", sep='\t')


t1 = time.time()
print('tempo:', t1 - t0)