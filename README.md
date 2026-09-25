# Estimación del costo de transformación en órdenes de fabricación

Proyecto de machine learning aplicado a órdenes de producción de cápsulas duras. El objetivo es estimar el costo unitario de transformación (`costo_transformacion`) a partir de características disponibles al especificar una orden.

## Estado del análisis

- Se preparó un análisis exploratorio en formato de artículo y en español en `costos_eda_hilo_conductor_redaccion.ipynb`.
- Se excluyeron las órdenes con volumen de lote no positivo y las que no registran insumos. La población analítica resultante contiene **529 órdenes**, **145 códigos de producto** y **20 periodos**.
- Las unidades son: duración en horas, costos en COP, consumo de insumos en kg y costo objetivo en COP por unidad producida. La unidad de `volumen_lote` no está especificada en los metadatos disponibles.
- El EDA describe la distribución del costo, asociaciones cuantitativas y diferencias por equipo, modalidad, número de insumos, categoría de producto y tipo de máquina. Las comparaciones entre categorías son descriptivas y no ajustadas; los grupos pequeños requieren cautela.
- El análisis evalúa Ridge bajo validación aleatoria repetida (órdenes de un catálogo conocido) y validación agrupada por `codigo_producto` (productos no observados durante el ajuste). La búsqueda de hiperparámetros y el preprocesamiento se realizan dentro de los pliegues.

## Resultados principales

La especificación B añade `tipo_equipo` y `num_insumos` a las variables de A. En la comparación validada, B obtiene R² medio de **0,509** en validación aleatoria y **0,317** en validación agrupada por producto. La prueba t corregida de Nadeau–Bengio, con ajuste de Holm, encuentra un aumento pequeño de R² estadísticamente significativo en la validación aleatoria (ΔR² = **0,026**, p ajustado = **0,0445**), pero no en la validación agrupada (ΔR² = **0,116**, p ajustado = **0,1796**). La adición fue conjunta: estos resultados no identifican el aporte individual de cada columna. La generalización a códigos de producto nuevos sigue siendo incierta.

El análisis por tipo de máquina y categoría de producto detecta diferencias globales en las distribuciones del costo. Estas asociaciones exploratorias no demuestran causalidad ni aíslan efectos entre variables correlacionadas.

## Modelo reutilizable

El pipeline Ridge de la especificación B se ajustó con las 529 órdenes. `alpha` se seleccionó con `GroupKFold` de cinco pliegues agrupado por `codigo_producto`, y el pipeline final se reajustó con toda la población elegible. El artefacto integra imputación, estandarización, codificación categórica y regresión Ridge.

- `outputs/modelo_ridge_costos/modelo_ridge_costo_transformacion.joblib`: pipeline ajustado.
- `outputs/modelo_ridge_costos/predecir_costos.py`: función y comando para producir predicciones.
- `outputs/modelo_ridge_costos/metadatos_modelo.json`: variables, filtros, hiperparámetro y versiones.
- `outputs/modelo_ridge_costos/README.md`: instrucciones específicas del artefacto.

El RMSE de los pliegues usados para seleccionar `alpha` es un resultado de selección y no una estimación externa imparcial. Para comparar este Ridge con otros algoritmos, deben usarse las métricas de validación anidada del notebook y particiones comunes.

### Generar predicciones

Desde la raíz del proyecto y con el entorno `project-env` activo, el CSV de entrada debe estar separado por punto y coma y contener estos predictores:

`codigo_producto`, `volumen_lote`, `equipo_id`, `categoria_producto`, `insumo_1`, `insumo_2`, `modalidad_proceso`, `tipo_equipo`, `num_insumos`.

```bash
python outputs/modelo_ridge_costos/predecir_costos.py ordenes_nuevas.csv predicciones.csv
```

La salida conserva las columnas de entrada y añade `pred_costo_transformacion_COP_unidad`.

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
