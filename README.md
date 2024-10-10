# Predicción de calidad de vinos

## Descripción del proyecto

Este proyecto tiene como objetivo predecir la calidad de los vinos utilizando técnicas de Machine Learning. Utilizamos un modelo de Random Forest para clasificar los vinos en buenos o malos basándonos en características químicas.

## Datos utilizados

El conjunto de datos utilizado proviene de [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). Dicho dataset contiene varias características químicas de diferentes vinos y su calidad evaluada en una escala de 0 a 10.

### Características del dataset

- `fixed acidity`: Acidez fija del vino.
- `volatile acidity`: Acidez volátil.
- `citric acid`: Nivel de ácido cítrico.
- `residual sugar`: Azúcar residual.
- `chlorides`: Concentración de cloruros.
- `free sulfur dioxide`: Dióxido de azufre libre.
- `total sulfur dioxide`: Dióxido de azufre total.
- `density`: Densidad del vino.
- `pH`: Medida del pH del vino.
- `sulphates`: Concentración de sulfatos.
- `alcohol`: Porcentaje de alcohol.
- `quality`: Calificación del vino.

## Técnicas de Machine Learning

Se implementaron las siguientes técnicas:

- **Modelo de Random Forest**: Para la clasificación de la calidad de los vinos.
- **Ajuste de Hiperparámetros**: Utilizando técnicas como Grid Search y Validación Cruzada.

## Resultados clave

El modelo alcanzó una precisión de **85%** y se identificaron las características más influyentes en la predicción de la calidad del vino.

## Instalación

1. Clona este repositorio: `git clone [https://github.com/hugoortuno/chemical-quality-wine-predictions]`.
2. Navega al directorio del proyecto: `cd [chemical-quality-wine-predictions]`.
3. Instala las dependencias: `pip install -r requirements.txt`.

## Ejecución del código

Para ejecutar el análisis, abre el notebook de Jupyter incluido en el repositorio.

## Conclusiones

Este proyecto demuestra cómo se pueden utilizar técnicas de Machine Learning para predecir la calidad de los vinos y resalta la importancia de la ingeniería de características en el proceso de modelado.

## Créditos

- [Hugo Ortuño Suárez]
