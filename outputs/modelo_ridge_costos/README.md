# Modelo Ridge reutilizable para costos

## Artefactos
- `modelo_ridge_costo_transformacion.joblib`: pipeline ajustado que incluye imputación, estandarización, codificación One-Hot y Ridge.
- `predecir_costos.py`: función `predecir(DataFrame)` y CLI para generar predicciones.
- `metadatos_modelo.json`: predictores, filtros, unidad de respuesta, alfa y versiones de software.

## Ajuste
El pipeline se ajustó con las 529 órdenes elegibles y 145 códigos de producto. La especificación B incluye `tipo_equipo` y `num_insumos`. El hiperparámetro Ridge seleccionado fue alpha = 3.16228. Alpha se seleccionó mediante GroupKFold de 5 pliegues, agrupado por código de producto, y luego GridSearchCV reajustó el pipeline con todos los datos.

La métrica de esta búsqueda se usa para seleccionar alpha. Para comparar algoritmos, deben usarse las métricas de validación anidada del notebook y particiones comunes. La evaluación agrupada por producto es la referencia más pertinente si se esperan códigos nuevos.

## Uso desde Python
```python
import pandas as pd
from predecir_costos import predecir

ordenes = pd.read_csv("ordenes_nuevas.csv", sep=";")
ordenes["pred_costo_transformacion_COP_unidad"] = predecir(ordenes)
```

## Uso desde terminal
```bash
python predecir_costos.py ordenes_nuevas.csv predicciones.csv
```
El archivo de entrada requiere estas columnas: `codigo_producto, volumen_lote, equipo_id, categoria_producto, insumo_1, insumo_2, modalidad_proceso, tipo_equipo, num_insumos`. La columna `volumen_lote` debe ser positiva y `num_insumos` mayor que cero, coherente con la población usada en el entrenamiento. La codificación de categorías no vistas se ignora de forma segura por el pipeline, pero su desempeño no queda garantizado por ello.
