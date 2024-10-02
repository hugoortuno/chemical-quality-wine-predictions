import os
from functions import (
    cargar_datos,
    limpiar_datos,
    ingenieria_caracteristicas,
    construir_modelo,
    calcular_kpis,
    exportar_datos
)

def main():
    # Cargar el conjunto de datos
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    datos_vino = cargar_datos(url)

    # Limpiar los datos
    datos_vino = limpiar_datos(datos_vino)

    # Ingeniar características
    datos_vino = ingenieria_caracteristicas(datos_vino)

    # Construir el modelo
    precision, reporte = construir_modelo(datos_vino)
    print(f"Precisión: {precision}\nReporte de Clasificación:\n{reporte}")

    # Calcular KPIs
    kpis = calcular_kpis(datos_vino)
    print(f"KPIs:\n{kpis}")

    # Crear la carpeta 'data' si no existe
    os.makedirs("data", exist_ok=True)

    # Exportar el DataFrame a CSV
    exportar_datos(datos_vino, "data/MLChemicalQualityWine.csv")

if __name__ == "__main__":
    main()
