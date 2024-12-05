import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funciones_principales.volumen_de_escorrentia import scs_runoff_volume
from funciones_principales.hidrograma_unitario_scs import scs_unit_hydrograph
from funciones_principales.coeficiente_de_nash_sutcliffe import calculate_nse

# Listas
results = []
simulated_hydrographs = []

# Ruta de los archivos de caudal y precipitación observadas
qobs_path = r"C:\Users\LENOVO\Downloads\Modelo_TR20\Caudales_Precipitaciones_13_10_07\Caudales observados.csv"
pobs_path = r"C:\Users\LENOVO\Downloads\Modelo_TR20\Caudales_Precipitaciones_13_10_07\Precipitaciones observadas.csv"
output_path = r"C:\Users\LENOVO\Downloads\Modelo_TR20\monte_carlo_results.csv"

# Leer datos de entrada
# Notas:
# 1. La fechas en el archivo csv deben estar en mm/dd/yy hh:mm
# 2. Los datos no deben contar con encabezado, es decir, las columnas
# 3. La primer columna debe almacenar los datos de tiempo, y la segunda columna los datos de caudal obervados o precipitación observada
qobs_data = pd.read_csv(qobs_path, sep=',', header=None,
                        names=['Datetime', 'Q_obs'])
pobs_data = pd.read_csv(pobs_path, sep=',', header=None,
                        names=['Datetime', 'P_obs'])

# Leer y formatear las fechas correctamente
# Nota:
# 1. Aquí pueden pasar problemas, cuando se vaya a tratar de leer las fechas desde el archivo csv. En todo caso y, si surge algun problema cuando se van a leer y formatear las fechas, se recomienda ajustar solo esta parte del codigo para subsanar el error en este sentido.
qobs_data['Datetime'] = pd.to_datetime(
    qobs_data['Datetime'], format='%m/%d/%Y %I:%M:%S %p'
)
pobs_data['Datetime'] = pd.to_datetime(
    pobs_data['Datetime'], format='%m/%d/%Y %I:%M:%S %p'
)

# Establecer 'Fecha y hora' como índice
qobs_data.set_index('Datetime', inplace=True)
pobs_data.set_index('Datetime', inplace=True)

# Asegurar que los índices tengan una frecuencia definida
qobs_data = qobs_data.asfreq('min')
pobs_data = pobs_data.asfreq('min')

# Reindexar para garantizar que ambos DataFrames tengan el mismo índice de tiempo
common_index = qobs_data.index.union(pobs_data.index)
qobs_data = qobs_data.reindex(common_index)
pobs_data = pobs_data.reindex(common_index)

# Manejar valores faltantes
pobs_data['P_obs'] = pd.to_numeric(pobs_data['P_obs'], errors='coerce')
pobs_data['P_obs'] = pobs_data['P_obs'].fillna(0)
qobs_data['Q_obs'] = qobs_data['Q_obs'].interpolate(method='time')

# Extraer valores
observed_flow = qobs_data['Q_obs'].values  # m³/s
rain_depth = pobs_data['P_obs'].values  # mm

# Pregunte al usuario por el número de simulaciones
num_simulations = int(input("Ingrese el número de simulaciones: "))

# Introducción del área (catchment_area) y paso del tiempo para el análisis (dt)
catchment_area = 0.41  # km2
dt = 60  # segundos

# Simulación por Montecarlo
for i in range(num_simulations):
    # Generar parámetros aleatorios
    CN = np.random.uniform(40, 90)
    # Lag_time en segundos (4 to 15 minutos)
    lag_time = np.random.uniform(4 * 60, 15 * 60)
    # lambda entre 0 and 1.0
    lambda_val = np.random.uniform(0.0, 1.0)

    # Calcular el volumen de escorrentía
    runoff_volume = scs_runoff_volume(
        rain_depth, CN, catchment_area, lambda_val)

    # Compruebe si runoff_volume no es todo ceros
    if np.all(runoff_volume == 0):
        nse = np.nan
        simulated = np.zeros_like(observed_flow)
    else:
        # Generar el hidrograma simulado
        simulated = scs_unit_hydrograph(runoff_volume, lag_time, dt)

        # Asegúrese de que el hidrograma simulado se alinee con los datos observados
        min_length = min(len(simulated), len(observed_flow))
        simulated = simulated[:min_length]
        observed_trimmed = observed_flow[:min_length]

        # Calcular NSE
        nse = calculate_nse(observed_trimmed, simulated)

    # Guardar resultados
    results.append({'Iteration': i + 1,
                    'Lambda': lambda_val,
                    'CN': CN,
                    'Lag_Time_min': lag_time / 60,
                    'NSE': nse})

    # Almacenar hidrograma simulado
    simulated_hydrographs.append(simulated)

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)

# Asegúrese de que la 'Iteración' sea de tipo entero
results_df['Iteration'] = results_df['Iteration'].astype(int)

# Guardar resultados en CSV
results_df.to_csv(output_path, index=False)

# Muestra las 5 mejores simulaciones con el mejor NSE
print("Top 5 simulaciones con mejor NSE:")
top5 = results_df.sort_values(by='NSE', ascending=False).head(5)
print(top5)

# Obtén el índice de la mejor simulación (ajuste para índice de base cero)
best_index = int(top5.iloc[0]['Iteration'] - 1)

# Recupera el mejor hidrograma simulado
best_simulated = simulated_hydrographs[best_index]

# Asegurar la alineación
min_length = min(len(best_simulated), len(observed_flow))
best_simulated = best_simulated[:min_length]
observed_trimmed = observed_flow[:min_length]

# Recuperar índice de tiempo
time_index = qobs_data.index[:min_length]

# Tabular el mejor hidrograma
hydrograph_df = pd.DataFrame({
    'Datetime': time_index,
    'Q_Observado_m3/s': observed_trimmed,
    'Q_Simulado_m3/s': best_simulated
})

# Creación del archivo donde estará el mejor hidrograma
hydrograph_output_path = r"C:\Users\LENOVO\Downloads\Modelo_TR20\mejor_hidrograma.csv"

# Guarde el hidrograma en CSV
hydrograph_df.to_csv(hydrograph_output_path, index=False)

print(f"La hidrograma de la mejor simulación se ha guardado en: {
      hydrograph_output_path}")

# Graficar
plt.figure(figsize=(12, 6))
plt.plot(time_index, hydrograph_df['Q_Observado_m3/s'],
         label="Q Observado (m3/s)", color="blue")
plt.plot(time_index, hydrograph_df['Q_Simulado_m3/s'],
         label="Q Simulado (L/s)", color="red", linestyle="solid")
plt.title("Hidrograma Simulado vs Observado - Mejor Simulación")
plt.xlabel("Tiempo (s)")
plt.ylabel("Caudal (m3/s)")
plt.legend()
plt.grid()
plt.show()
