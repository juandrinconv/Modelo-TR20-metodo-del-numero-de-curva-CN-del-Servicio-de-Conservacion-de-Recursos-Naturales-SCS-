# 1. Descripción general
Este es un código que resuelve el modelo TR20 del método del Número de Curva (CN) del Servicio de Conservación de Recursos Naturales (SCS), con el fin de generar el hidrograma a interés, de hecho, se provee la calibración del modelo matemático a través del método de MonteCarlo.

Cabe mencionar, que dicho código está acondiconado a un evento de precipitación que tiene datos cada minuto, por tanto, para llevar a cabo experimentos fuera de dicha característica se deberá revisar este código y, si es necesario, ajustarlo para reproducir acertadamente el fenómeno en cuestión.

En todo caso y, para que el código se ajuste a las necesidades que requiera cada persona, se recomienda clonar el código en cuestión y revisarlo minuciosamente.

# 2. Descripción de las funciones principales del código

## 2.1 scs_runoff_volume
Calcula la escorrentía directa (exceso de lluvia) en una cuenca utilizando el método del Número de Curva (CN) del Servicio de Conservación de Recursos Naturales (SCS)
        
### 2.1.1 Parámetros de entrada

#### 2.1.1.1 rain_depth (milimetros) [mm]
- Tipo: Lista o arreglo de valores numéricos.
- Significado: Serie de precipitaciones (tener en cuenta el intervalo de tiempo de medición de la precipitación)
- Rol: Proporciona los datos de entrada necesarios para calcular el exceso de precipitación.

#### 2.1.1.2 CN (adimensional)
- Tipo: Número (valor único).
- Significado: Número de curva del SCS, que indica las características de escorrentía de la cuenca.
- Rol: Determina la capacidad de infiltración del suelo. Valores más altos indican mayor escorrentía y menor infiltración.

#### 2.1.1.3 catchment_area (kilómetros cuadrados) [km2]
- Tipo: Número (valor único).
- Significado: Área de la cuenca.
Rol: Permite convertir el exceso de precipitación (en mm) a volumen de escorrentía (en m³).

#### 2.1.1.4 lambda_val (adimensional)
- Tipo: Número (valor único, típicamente varía entre 0 y 1).
- Significado: Fracción de la capacidad potencial de retención inicial. Por defecto en el método del SCS es 0.2.
- Rol: Define la fracción del almacenamiento total inicial usado para calcular la precipitación inicial abstraída.

### 2.1.2 Variables internas

#### 2.1.2.1 S (milimetros) [mm]
- Significado: Capacidad de retención máxima del suelo.
- Rol: Representa la capacidad máxima de almacenamiento de agua del suelo antes de que ocurra escorrentía.

#### 2.1.2.2 Ia (milimetros) [mm]
- Significado: Abstracción inicial.
- Rol: Representa la cantidad de lluvia necesaria para superar las pérdidas iniciales (infiltración, almacenamiento superficial).

### 2.1.3 Salida de la función

#### 2.1.3.1 incremental_excess (volumen) [m³] [volumen para cada paso de tiempo, según el intervalo de tiempo en la cual venga la lluvia rain_depth]
- Tipo: Arreglo de valores numéricos.
- Significado: Escorrentía directa incremental en m³ para cada paso de tiempo.

## 2.2 scs_unit_hydrograph
Esta función implementa el método del hidrograma unitario del SCS adimensional para calcular el caudal resultante (hidrograma) a partir del volumen de escorrentía incremental, aplicando el concepto de retardo del flujo en la cuenca.

### 2.2.1 Parámetros de entrada

#### 2.2.1.1 runoff_volume (volumen) [m³] [volumen para cada paso de tiempo, según el intervalo de tiempo en la cual venga la lluvia rain_depth, la cual se introduce en scs_runoff_volume, es decir, en la función descrita en el apartado (2.1)]
- Tipo: Lista o arreglo de valores numéricos.
- Significado: Volumen de escorrentía incremental. Es lo que devuelve la función del apartado (1.1), es decir, la función scs_runoff_volume.
- Rol: Representa el agua que entra al sistema en cada paso de tiempo para ser ruteada.
    
#### 2.2.1.2 dt (segundos) [s] [debe ser igual al intervalo de tiempo en el cual se registró la lluvia rain_depth, la cual se introduce en scs_runoff_volume, es decir, en la función descrita en el apartado (2.1)]
- Tipo: Número (valor único).
- Significado: Paso de tiempo del análisis.
- Rol: Define la resolución temporal del cálculo.
            
#### 2.2.1.3 lag_time (segundos) [s]
- Tipo: Número (valor único).
- Significado: Tiempo de retardo de la cuenca.
- Rol: Define el tiempo característico de respuesta de la cuenca, es decir, el tiempo que tarda el flujo en alcanzar el punto de control después de un evento de lluvia.
        
### 2.2.2 Salida de la función

#### 2.2.2.1 discharge_rate 
- Tipo: Arreglo de valores numéricos.
- Significado: Hidrograma de salida en términos de caudal versus el tiempo (si se adoptan los datos en las unidades recomendadas, el caudal estará en m3/s y el tiempo en minutos)

# 3. Bibliotecas utilizadas
La información sobre las bibliotecas empleadas se encuetran en el archivo requirements.txt

# 4. Versión de Python utilizada
Python 3.12.4

# 5. Información para cuando se vaya a clonar el repositorio 
Este repositorio se subió con el entorno virtual por defecto de Python, por ende, no es necesario crear por separado dicho entorno antes de clonar el repositorio, sino que simplemente, después de clonar el repositorio, se instalan todas las dependencias en el entorno virtual, de hecho, no es ocioso mencionar que toda la información sobre las dependencias a instalar se encuentran en el archivo requirements.txt

# 6. Información sobre los dos ejemplos que se pueden ensayar
Existe la información de dos eventos en las carpetas llamadas "Caudales_Precipitaciones_08_10_07" y "Caudales_Precipitaciones_13_10_07", las cuales contienen el registro de caudales y precipitación observada en el campus de la Universidad Nacional de Colombia Sede Bogotá.

En este sentido y, utilizando la información suministrada, se pueden correr perfectamente los dos modelos correspondientes a cada evento, con el fin de que la persona interesada puede apreciar que el código funciona.