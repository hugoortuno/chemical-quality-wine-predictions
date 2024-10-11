import os
from functions import (
    cargar_datos,
    limpiar_datos,
    ingenieria_caracteristicas,
    construir_modelo,
    calcular_kpis,
    exportar_datos,
    graficar_distribucion_calidad,
    graficar_alcohol_vs_calidad,
    graficar_acidez_vs_calidad
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

    # Graficar y guardar imágenes
    graficar_distribucion_calidad(datos_vino, "data/distribucion_calidad.png")
    graficar_alcohol_vs_calidad(datos_vino, "data/alcohol_vs_calidad.png")
    graficar_acidez_vs_calidad(datos_vino, "data/acidez_vs_calidad.png")

    # Exportar el DataFrame a CSV
    exportar_datos(datos_vino, "data/MLChemicalQualityWine.csv")

if __name__ == "__main__":
    main()
