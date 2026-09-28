# Estimación del costo de transformación en órdenes de fabricación

Proyecto de machine learning aplicado a órdenes de producción de cápsulas duras. El objetivo es estimar el costo unitario de transformación (`costo_transformacion`) a partir de características disponibles al especificar una orden.

## Estado del análisis

- Se preparó un análisis exploratorio en formato de artículo y en español en `costos_eda_hilo_conductor_redaccion.ipynb`.
- Se excluyeron las órdenes con volumen de lote no positivo y las que no registran insumos. La población elegible contiene **529 órdenes** y **145 códigos de producto**. Se reserva aleatoriamente por fila una prueba final de 106 órdenes; el desarrollo utiliza 423 órdenes. 47 de los 60 códigos de prueba también aparecen en entrenamiento; por azar, algunos códigos solo aparecen en prueba.
- Las unidades son: duración en horas, costos en COP, consumo de insumos en kg y costo objetivo en COP por unidad producida. La unidad de `volumen_lote` no está especificada en los metadatos disponibles.
- El EDA describe la distribución del costo, asociaciones cuantitativas y diferencias por equipo, modalidad, número de insumos, categoría de producto y tipo de máquina. Se adopta un supuesto estacionario y se excluye `periodo` del análisis y del modelado. Las comparaciones entre categorías son descriptivas y no ajustadas; los grupos pequeños requieren cautela.
- El análisis evalúa Ridge bajo validación aleatoria repetida (órdenes de un catálogo conocido) y validación agrupada por `codigo_producto` (productos no observados durante el ajuste). La búsqueda de hiperparámetros y el preprocesamiento se realizan dentro de los pliegues.

## Resultados principales

La especificación B añade `tipo_equipo` y `num_insumos` a las variables de A. En validación aleatoria repetida, que aproxima predicciones para nuevas órdenes de productos conocidos, Ridge B obtiene RMSE **10,75 ± 0,98**, MAE **7,50 ± 0,53 COP/unidad** y R² **0,462 ± 0,09**. Ridge A alcanza RMSE 11,08 ± 0,99, MAE 7,66 ± 0,52 y R² 0,431 ± 0,07; la mejora de R² no es significativa tras la corrección de Nadeau–Bengio y Holm (p = **0,166**). En validación agrupada, B obtiene R² medio **0,261** frente a 0,185 para A, sin mejora estadísticamente significativa (p de Holm = **0,191**). En la prueba aleatoria final, Ridge B logra RMSE **11,00**, MAE **7,69 COP/unidad** y R² **0,539**; el predictor de la media registra RMSE 16,19. Como la prueba mezcla órdenes de productos compartidos y códigos ausentes de entrenamiento, no aísla ninguno de esos dos escenarios; la validación agrupada aporta una estimación complementaria para productos no vistos.

El análisis por tipo de máquina y categoría de producto detecta diferencias globales en las distribuciones del costo. Estas asociaciones exploratorias no demuestran causalidad ni aíslan efectos entre variables correlacionadas.

## Modelo reutilizable

El pipeline Ridge de la especificación B ya está exportado y se carga para reutilizarlo; la sección de exportación del notebook no vuelve a ajustar el modelo. El artefacto existente fue entrenado con las 529 órdenes elegibles. `alpha` se seleccionó con `GroupKFold` de cinco pliegues agrupado por `codigo_producto`, y el pipeline final se reajustó con toda la población elegible. El artefacto integra imputación, estandarización, codificación categórica y regresión Ridge.

- `outputs/modelo_ridge_costos/modelo_ridge_costo_transformacion.joblib`: pipeline ajustado.
- `outputs/modelo_ridge_costos/predecir_costos.py`: función y comando para producir predicciones.
- `outputs/modelo_ridge_costos/metadatos_modelo.json`: variables, filtros, hiperparámetro y versiones.
- `outputs/modelo_ridge_costos/README.md`: instrucciones específicas del artefacto.

El artefacto de uso futuro se reajustó previamente con las 529 órdenes, incluida la muestra reservada. Por ello, las métricas de prueba del notebook corresponden al Ridge entrenado solo con el conjunto de desarrollo y no son una evaluación independiente del artefacto final. El RMSE de los pliegues usados para seleccionar `alpha` es un resultado de selección y no una estimación externa imparcial. Para comparar este Ridge con otros algoritmos, deben usarse las métricas de validación anidada del notebook y particiones comunes.

### Generar predicciones

Desde la raíz del proyecto y con el entorno `project-env` activo, el CSV de entrada debe estar separado por punto y coma y contener estos predictores:

`codigo_producto`, `volumen_lote`, `equipo_id`, `categoria_producto`, `insumo_1`, `insumo_2`, `modalidad_proceso`, `tipo_equipo`, `num_insumos`.

```bash
python outputs/modelo_ridge_costos/predecir_costos.py ordenes_nuevas.csv predicciones.csv
```

La salida conserva las columnas de entrada y añade `pred_costo_transformacion_COP_unidad`.
El notebook incluye una celda interactiva que solicita una o varias órdenes en formato JSON y devuelve sus predicciones usando el artefacto exportado.

## Estructura

```text
.
├── costos_eda_hilo_conductor_redaccion.ipynb
├── data_costos.csv
├── outputs/
│   └── modelo_ridge_costos/
│       ├── modelo_ridge_costo_transformacion.joblib
│       ├── metadatos_modelo.json
│       ├── predecir_costos.py
│       └── README.md
└── README.md
```

## Reproducibilidad

El análisis y la exportación se prepararon para el entorno Conda `project-env`. Las versiones de Python, scikit-learn y joblib usadas al exportar el pipeline están registradas en `metadatos_modelo.json`. La serialización con joblib debe cargarse en un entorno compatible con esas versiones.
