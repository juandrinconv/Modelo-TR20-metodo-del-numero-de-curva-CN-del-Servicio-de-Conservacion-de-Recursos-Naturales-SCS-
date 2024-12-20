import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve
# Implantación de semilla para que se compartan los mismos números aleatorios de los parámetros en ambos eventos
np.random.seed(40)
# Hidrograma unitario
SCS_DIMENSIONLESS_UH = np.array([
    [0.0, 0.000],
    [0.1, 0.030],
    [0.2, 0.100],
    [0.3, 0.190],
    [0.4, 0.310],
    [0.5, 0.470],
    [0.6, 0.660],
    [0.7, 0.820],
    [0.8, 0.930],
    [0.9, 0.990],
    [1.0, 1.000],
    [1.1, 0.990],
    [1.2, 0.930],
    [1.3, 0.860],
    [1.4, 0.780],
    [1.5, 0.680],
    [1.6, 0.560],
    [1.7, 0.460],
    [1.8, 0.390],
    [1.9, 0.330],
    [2.0, 0.280],
    [2.2, 0.207],
    [2.4, 0.147],
    [2.6, 0.107],
    [2.8, 0.077],
    [3.0, 0.055],
    [3.2, 0.040],
    [3.4, 0.029],
    [3.6, 0.021],
    [3.8, 0.015],
    [4.0, 0.011],
    [4.5, 0.005],
    [5.0, 0.000]
])


def scs_runoff_volume(rain_depth, CN, catchment_area, lambda_val):
    """
    Compute the runoff volume using the SCS method.
    """
    S = (25400. - 254. * CN) / CN  # en mm
    Ia = lambda_val * S  # en mm

    # Precipitaciones acumuladas
    cumulative_rain_depth = np.cumsum(rain_depth)

    Pe_cumulative = np.where(cumulative_rain_depth > Ia,
                             ((cumulative_rain_depth - Ia) ** 2) /
                             (cumulative_rain_depth - Ia + S),
                             0)
    # Exceso de precipitación incremental
    incremental_Pe = np.diff(Pe_cumulative, prepend=0)

    # Exceso incremental en metros cúbicos por paso de tiempo
    incremental_excess = incremental_Pe * catchment_area * 1e6 / 1000  # mm a m³
    return incremental_excess


def scs_unit_hydrograph(runoff_volume, lag_time, dt):
    """
    Route incremental runoff through SCS Unit Hydrograph.
    """
    num_points = int(5 * lag_time / dt)
    num_points = max(num_points, 1)  # Asegúrese de tener al menos un punto
    t_d = np.linspace(0, 5, num_points)
    scs_q_curve = interp1d(SCS_DIMENSIONLESS_UH[:, 0],
                           SCS_DIMENSIONLESS_UH[:, 1],
                           kind='linear',
                           fill_value="extrapolate",
                           bounds_error=False)
    scs2 = scs_q_curve(t_d)
    # Normalizar el hidrograma unitario
    area = np.trapezoid(scs2, t_d)
    scs2 /= area
    # Escala scs2 para tener unidades consistentes
    scs2 /= lag_time
    # Metodológia para realizar la Convolución
    discharge_rate = convolve(runoff_volume, scs2, mode="full")[
        :len(runoff_volume)]
    return discharge_rate


def calculate_nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)


# Inicial configuración
qobs_path_1 = r"C:\Users\Hernan Moreno\Desktop\Informe final\Qobs08.csv"
qobs_path_2 = r"C:\Users\Hernan Moreno\Desktop\Informe final\Qobs13.csv"
pobs_path_1 = r"C:\Users\Hernan Moreno\Desktop\Informe final\Pobs08.csv"
pobs_path_2 = r"C:\Users\Hernan Moreno\Desktop\Informe final\Pobs13.csv"
output_path_1 = r"C:\Users\Hernan Moreno\Desktop\Informe final\monte_carlo_results_08.csv"
output_path_2 = r"C:\Users\Hernan Moreno\Desktop\Informe final\monte_carlo_results_13.csv"
hydrograph_output_path_1 = r"C:\Users\Hernan Moreno\Desktop\Informe final\mejor_hidrograma_08.csv"
hydrograph_output_path_2 = r"C:\Users\Hernan Moreno\Desktop\Informe final\mejor_hidrograma_13.csv"
area_km2 = 0.41
dt = 60  # Paso de tiempo en segundos (1 minuto)
# lectura de datos de entrada
qobs_data_1 = pd.read_csv(qobs_path_1, sep=',',
                          header=None, names=['Datetime', 'Q_obs'])
pobs_data_1 = pd.read_csv(pobs_path_1, sep=',',
                          header=None, names=['Datetime', 'P_obs'])
qobs_data_2 = pd.read_csv(qobs_path_2, sep=',',
                          header=None, names=['Datetime', 'Q_obs'])
pobs_data_2 = pd.read_csv(pobs_path_2, sep=',',
                          header=None, names=['Datetime', 'P_obs'])
# Asegúrese de que las fechas estén formateadas correctamente en los CVS
qobs_data_1['Datetime'] = pd.to_datetime(
    qobs_data_1['Datetime'], format='%m/%d/%Y %H:%M')
pobs_data_1['Datetime'] = pd.to_datetime(
    pobs_data_1['Datetime'], format='%m/%d/%Y %H:%M')
qobs_data_2['Datetime'] = pd.to_datetime(
    qobs_data_2['Datetime'], format='%m/%d/%Y  %H:%M:%S ')
pobs_data_2['Datetime'] = pd.to_datetime(
    pobs_data_2['Datetime'], format='%m/%d/%Y  %H:%M:%S ')
# Establecer 'Datetime' como índice
qobs_data_1.set_index('Datetime', inplace=True)
pobs_data_1.set_index('Datetime', inplace=True)
qobs_data_2.set_index('Datetime', inplace=True)
pobs_data_2.set_index('Datetime', inplace=True)
# Asegúrese de que los índices tengan una frecuencia definida
qobs_data_1 = qobs_data_1.asfreq('min')
pobs_data_1 = pobs_data_1.asfreq('min')
qobs_data_2 = qobs_data_2.asfreq('min')
pobs_data_2 = pobs_data_2.asfreq('min')
# Reindexar para garantizar que ambos DataFrames tengan el mismo índice de tiempo
common_index_1 = qobs_data_1.index.union(pobs_data_1.index)
common_index_2 = qobs_data_2.index.union(pobs_data_2.index)
qobs_data_1 = qobs_data_1.reindex(common_index_1)
pobs_data_1 = pobs_data_1.reindex(common_index_1)
qobs_data_2 = qobs_data_2.reindex(common_index_2)
pobs_data_2 = pobs_data_2.reindex(common_index_2)
# Manejar valores faltantes
pobs_data_1['P_obs'] = pd.to_numeric(pobs_data_1['P_obs'], errors='coerce')
pobs_data_1['P_obs'].fillna(0, inplace=True)
qobs_data_1['Q_obs'].interpolate(method='time', inplace=True)
pobs_data_2['P_obs'] = pd.to_numeric(pobs_data_2['P_obs'], errors='coerce')
pobs_data_2['P_obs'].fillna(0, inplace=True)
qobs_data_2['Q_obs'].interpolate(method='time', inplace=True)
# Extracción de valores
rain_depth_1 = pobs_data_1['P_obs'].values  # en mm
observed_flow_1 = qobs_data_1['Q_obs'].values  # en L/s
rain_depth_2 = pobs_data_2['P_obs'].values  # en mm
observed_flow_2 = qobs_data_2['Q_obs'].values  # en L/s
# Convertir el caudal observado de L/s a m³/s

observed_flow_1_m3s = observed_flow_1 / 1000  # L/s a m³/s
observed_flow_2_m3s = observed_flow_2 / 1000  # L/s a m³/s
# Preguntar el número de simulaciones que se desea ejecutar
num_simulations = int(input("Ingrese el número de simulaciones: "))

results_1 = []
# Para almacenar los hidrogramas simulados para el primer conjunto de datos (08/10/2007)
simulated_hydrographs_1 = []

results_2 = []
# Para almacenar los hidrogramas simulados para el segundo conjunto de datos (13/10/2007)
simulated_hydrographs_2 = []

# Simulación de montecarlo para ambos eventos bajo la condición de tener los mismos parametros

for i in range(num_simulations):
    # generación de parámetros aleatorios

    CN = np.random.uniform(40, 95)  # Ajuste del rango para el número de curva
    lag_time = np.random.uniform(5 * 60, 15 * 60)  # rango de Tlag en minutos
    lambda_val = np.random.uniform(0.0, 1.0)  # Lambda entre 0 y 1

    # Calcular el volumen de escorrentía para el primer evento 08/10/2007
    runoff_volume_1 = scs_runoff_volume(rain_depth_1, CN, area_km2, lambda_val)

    # Comprueba si runoff_volume no está compuesto únicamente de ceros
    if np.all(runoff_volume_1 == 0):
        nse_1 = np.nan
        simulated_1 = np.zeros_like(observed_flow_1_m3s)
    else:
        # Generar hidrograma primer evento
        simulated_1 = scs_unit_hydrograph(runoff_volume_1, lag_time, dt)

        # Asegúrese de que el hidrograma simulado se alinee con los datos observados
        min_length_1 = min(len(simulated_1), len(observed_flow_1_m3s))
        simulated_1 = simulated_1[:min_length_1]
        observed_trimmed_1 = observed_flow_1_m3s[:min_length_1]

        # calculo del NSE para el primer evento
        nse_1 = calculate_nse(observed_trimmed_1, simulated_1)

    # Almacenar resultados para el primer conjunto de datos
    results_1.append({'Iteration': i + 1,
                      'Lambda': lambda_val,
                      'CN': CN,
                      'Lag_Time_min': lag_time / 60,
                      'NSE': nse_1})

    # Almacenar el hidrograma simulado para el primer conjunto de datos
    simulated_hydrographs_1.append(simulated_1)

    # Mismos cálculos pero ahora para el evento dos (13/10/2007)
    runoff_volume_2 = scs_runoff_volume(rain_depth_2, CN, area_km2, lambda_val)
    if np.all(runoff_volume_2 == 0):
        nse_2 = np.nan
        simulated_2 = np.zeros_like(observed_flow_2_m3s)
    else:
        simulated_2 = scs_unit_hydrograph(runoff_volume_2, lag_time, dt)
        min_length_2 = min(len(simulated_2), len(observed_flow_2_m3s))
        simulated_2 = simulated_2[:min_length_2]
        observed_trimmed_2 = observed_flow_2_m3s[:min_length_2]
        nse_2 = calculate_nse(observed_trimmed_2, simulated_2)

    results_2.append({'Iteration': i + 1,
                      'Lambda': lambda_val,
                      'CN': CN,
                      'Lag_Time_min': lag_time / 60,
                      'NSE': nse_2})
    simulated_hydrographs_2.append(simulated_2)

results_df_1 = pd.DataFrame(results_1)
results_df_2 = pd.DataFrame(results_2)

# Asegúrese de que 'Iteración' sea de tipo entero
results_df_1['Iteration'] = results_df_1['Iteration'].astype(int)
results_df_2['Iteration'] = results_df_2['Iteration'].astype(int)

# Guardar resultados en formato CSV para el primer conjunto de datos
results_df_1.to_csv(output_path_1, index=False)
results_df_2.to_csv(output_path_2, index=False)

# Mostrar las 5 mejores simulaciones con mejor NSE para el primer conjunto de datos
print("Top 5 simulaciones con mejor NSE para el primer conjunto de datos:")
top5_1 = results_df_1.sort_values(by='NSE', ascending=False).head(5)
print(top5_1)


# Obtenga el índice de la mejor simulación para el primer conjunto de datos
# Ajuste para índice basado en cero
best_index_1 = int(top5_1.iloc[0]['Iteration'] - 1)

# Recupere el mejor hidrograma simulado para el primer conjunto de datos
best_simulated_1 = simulated_hydrographs_1[best_index_1]

# Asegurar la alineación
min_length_1 = min(len(best_simulated_1), len(observed_flow_1_m3s))
best_simulated_1 = best_simulated_1[:min_length_1]
observed_trimmed_1 = observed_flow_1_m3s[:min_length_1]

# Convertir el flujo simulado de m³/s a L/s para representar gráficamente el primer conjunto de datos
best_simulated_1_Ls = best_simulated_1 * 1000  # m³/s a L/s

# Recuperar el índice de tiempo para el primer conjunto de datos
time_index_1 = qobs_data_1.index[:min_length_1]

# Tabular el mejor hidrograma para el primer conjunto de datos
hydrograph_df_1 = pd.DataFrame({
    'Datetime': time_index_1,
    'Q_Observado_Ls': observed_trimmed_1 * 1000,  # m³/s a L/s
    'Q_Simulado_Ls': best_simulated_1_Ls
})

# Guardar el hidrograma en formato CSV para el primer conjunto de datos
hydrograph_df_1.to_csv(hydrograph_output_path_1, index=False)

print(f"La hidrograma de la mejor simulación para el primer conjunto de datos se ha guardado en: {
      hydrograph_output_path_1}")

# Trazado para el primer conjunto de datos
plt.figure(figsize=(12, 6))
plt.plot(time_index_1, hydrograph_df_1['Q_Observado_Ls'],
         label="Q Observado (L/s)", color="green")
plt.plot(time_index_1, hydrograph_df_1['Q_Simulado_Ls'],
         label="Q Simulado (L/s)", color="red", linestyle="--")
plt.title(
    "Hidrograma Simulado vs Observado - Mejor Simulación (Primer Conjunto de Datos)")
plt.xlabel("Tiempo")
plt.ylabel("Caudal (L/s)")
plt.legend()
plt.grid()
plt.show()

# Repita el proceso para el segundo conjunto de datos.

print("Top 5 simulaciones con mejor NSE para el segundo conjunto de datos:")
top5_2 = results_df_2.sort_values(by='NSE', ascending=False).head(5)
print(top5_2)

best_index_2 = int(top5_2.iloc[0]['Iteration'] - 1)
best_simulated_2 = simulated_hydrographs_2[best_index_2]

min_length_2 = min(len(best_simulated_2), len(observed_flow_2_m3s))
best_simulated_2 = best_simulated_2[:min_length_2]
observed_trimmed_2 = observed_flow_2_m3s[:min_length_2]

best_simulated_2_Ls = best_simulated_2 * 1000

time_index_2 = qobs_data_2.index[:min_length_2]

hydrograph_df_2 = pd.DataFrame({
    'Datetime': time_index_2,
    'Q_Observado_Ls': observed_trimmed_2 * 1000,
    'Q_Simulado_Ls': best_simulated_2_Ls
})

hydrograph_df_2.to_csv(hydrograph_output_path_2, index=False)

print(f"La hidrograma de la mejor simulación para el segundo conjunto de datos se ha guardado en: {
      hydrograph_output_path_2}")

plt.figure(figsize=(12, 6))
plt.plot(time_index_2, hydrograph_df_2['Q_Observado_Ls'],
         label="Q Observado (L/s)", color="green")
plt.plot(time_index_2, hydrograph_df_2['Q_Simulado_Ls'],
         label="Q Simulado (L/s)", color="red", linestyle="--")
plt.title(
    "Hidrograma Simulado vs Observado - Mejor Simulación (Segundo Conjunto de Datos)")
plt.xlabel("Tiempo")
plt.ylabel("Caudal (L/s)")
plt.legend()
plt.grid()
plt.show()


def calculate_nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)


# Filtrar las simulaciones con un NSE mayor a 0.75
good_simulations_1 = [result for result in results_1 if result['NSE'] > 0.75]
good_simulations_2 = [result for result in results_2 if result['NSE'] > 0.75]

# Almacenar los hidrogramas correspondientes a las simulaciones buenas
good_hydrographs_1 = [simulated_hydrographs_1[int(
    result['Iteration'] - 1)] for result in good_simulations_1]
good_hydrographs_2 = [simulated_hydrographs_2[int(
    result['Iteration'] - 1)] for result in good_simulations_2]

# Convertir las listas de hidrogramas a arrays para poder calcular percentiles
good_hydrographs_1 = np.array(good_hydrographs_1)
good_hydrographs_2 = np.array(good_hydrographs_2)

# Calcular los percentiles 5% y 95% para los hidrogramas simulados
percentile_5_1 = np.percentile(good_hydrographs_1, 5, axis=0)
percentile_95_1 = np.percentile(good_hydrographs_1, 95, axis=0)

percentile_5_2 = np.percentile(good_hydrographs_2, 5, axis=0)
percentile_95_2 = np.percentile(good_hydrographs_2, 95, axis=0)

# Asegúrate de que ya se han cargado los hidrogramas y los datos correctamente

# Seleccionar el mejor hidrograma simulado (con el mayor NSE)
best_simulated_1 = good_hydrographs_1[np.argmax(
    [result['NSE'] for result in good_simulations_1])]
best_simulated_2 = good_hydrographs_2[np.argmax(
    [result['NSE'] for result in good_simulations_2])]

# Generar gráfico para el primer conjunto de datos con banda de incertidumbre y los simulados
plt.figure(figsize=(12, 6))
plt.plot(time_index_1, observed_trimmed_1 * 1000,
         label="Q Observado (L/s)", color="blue")
plt.plot(time_index_1, best_simulated_1 * 1000,
         label="Q Simulado Mejor (L/s)", color="orange", linestyle="--")
plt.plot(time_index_1, percentile_5_1 * 1000,
         label="Percentil 5% (Simulado)", color="green", linestyle="--")
plt.plot(time_index_1, percentile_95_1 * 1000,
         label="Percentil 95% (Simulado)", color="green", linestyle="--")
plt.fill_between(time_index_1, percentile_5_1 * 1000,
                 percentile_95_1 * 1000, color="green", alpha=0.3)
plt.title("Análisis de Incertidumbre - Mejor Simulación (Primer Conjunto de Datos)")
plt.xlabel("Tiempo")
plt.ylabel("Caudal (L/s)")
plt.legend()
plt.grid()
plt.show()

# Generar gráfico para el segundo conjunto de datos con banda de incertidumbre y los simulados
plt.figure(figsize=(12, 6))
plt.plot(time_index_2, observed_trimmed_2 * 1000,
         label="Q Observado (L/s)", color="blue")
plt.plot(time_index_2, best_simulated_2 * 1000,
         label="Q Simulado Mejor (L/s)", color="orange", linestyle="--")
plt.plot(time_index_2, percentile_5_2 * 1000,
         label="Percentil 5% (Simulado)", color="green", linestyle="--")
plt.plot(time_index_2, percentile_95_2 * 1000,
         label="Percentil 95% (Simulado)", color="green", linestyle="--")
plt.fill_between(time_index_2, percentile_5_2 * 1000,
                 percentile_95_2 * 1000, color="green", alpha=0.3)
plt.title("Análisis de Incertidumbre - Mejor Simulación (Segundo Conjunto de Datos)")
plt.xlabel("Tiempo")
plt.ylabel("Caudal (L/s)")
plt.legend()
plt.grid()
plt.show()

# Guardar los resultados de la simulación con banda de incertidumbre para el primer conjunto de datos
uncertainty_hydrograph_1 = pd.DataFrame({
    'Datetime': time_index_1,
    'Q_Observado_Ls': observed_trimmed_1 * 1000,
    'Percentil_5_Ls': percentile_5_1 * 1000,
    'Percentil_95_Ls': percentile_95_1 * 1000
})
uncertainty_hydrograph_1.to_csv("uncertainty_hydrograph_1.csv", index=False)

# Guardar los resultados de la simulación con banda de incertidumbre para el segundo conjunto de datos
uncertainty_hydrograph_2 = pd.DataFrame({
    'Datetime': time_index_2,
    'Q_Observado_Ls': observed_trimmed_2 * 1000,
    'Percentil_5_Ls': percentile_5_2 * 1000,
    'Percentil_95_Ls': percentile_95_2 * 1000
})
uncertainty_hydrograph_2.to_csv("uncertainty_hydrograph_2.csv", index=False)
