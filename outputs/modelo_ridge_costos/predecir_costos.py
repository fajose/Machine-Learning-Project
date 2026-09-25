"""Predicción con el modelo Ridge exportado para costo unitario de transformación."""
from pathlib import Path
import argparse
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "modelo_ridge_costo_transformacion.joblib"
FEATURES = [
    "codigo_producto", "volumen_lote", "equipo_id", "categoria_producto",
    "insumo_1", "insumo_2", "modalidad_proceso", "tipo_equipo", "num_insumos"
]

def cargar_modelo():
    return joblib.load(MODEL_PATH)

def predecir(ordenes: pd.DataFrame) -> pd.Series:
    """Recibe un DataFrame con las nueve columnas predictoras y devuelve COP/unidad."""
    faltantes = [col for col in FEATURES if col not in ordenes.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")
    pred = cargar_modelo().predict(ordenes[FEATURES])
    return pd.Series(pred, index=ordenes.index, name="pred_costo_transformacion_COP_unidad")

def main():
    parser = argparse.ArgumentParser(description="Genera predicciones de costo unitario (COP/unidad).")
    parser.add_argument("entrada", help="CSV separado por punto y coma con las variables predictoras")
    parser.add_argument("salida", help="Ruta del CSV de salida")
    args = parser.parse_args()
    datos = pd.read_csv(args.entrada, sep=";")
    datos["pred_costo_transformacion_COP_unidad"] = predecir(datos)
    datos.to_csv(args.salida, sep=";", index=False)

if __name__ == "__main__":
    main()
