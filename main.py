import pandas as pd
from functions import LimpiarDatos, IngeniarCaracteristicas, ConstruirModelo, CalcularKPIs, ExportarResultados

# Cargar datos
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DatosVino = pd.read_csv(url, sep=';')

# Proceso del flujo
DatosVino = LimpiarDatos(DatosVino)
DatosVino = IngeniarCaracteristicas(DatosVino)
Precision, Reporte = ConstruirModelo(DatosVino)
KPIs = CalcularKPIs(DatosVino)

# Exportar resultados
ExportarResultados(DatosVino, "data/MLChemicalQualityWine.csv")
