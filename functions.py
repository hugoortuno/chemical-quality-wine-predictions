import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def cargar_datos(url):
    """Cargar el conjunto de datos desde una URL."""
    return pd.read_csv(url, sep=';')

def limpiar_datos(df):
    """Limpiar datos, eliminar nulos y asegurar que 'Calidad' sea un entero."""
    if df.isnull().sum().any():
        df = df.dropna()
    df['Calidad'] = df['quality'].astype(int)
    return df

def ingenieria_caracteristicas(df):
    """Agregar características derivadas al DataFrame."""
    df['EsBuena'] = (df['Calidad'] >= 6).astype(int)  # Calidad buena
    return df

def construir_modelo(df):
    """Construir y entrenar un modelo de Random Forest."""
    X = df.drop(columns=['quality', 'Calidad', 'EsBuena'])
    y = df['EsBuena']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    return precision, classification_report(y_test, y_pred)

def calcular_kpis(df):
    """Calcular KPIs basados en el DataFrame."""
    kpi_dict = {
        "Promedio de alcohol": df[df['EsBuena'] == 1]['alcohol'].mean(),
        "Promedio de acidez": df[df['EsBuena'] == 1]['volatile acidity'].mean(),
        "Número de vinos buenos": df[df['EsBuena'] == 1].shape[0],
        "Proporción de vinos buenos": df[df['EsBuena'] == 1].shape[0] / df.shape[0],
    }
    return kpi_dict

def exportar_datos(df, filename):
    """Exportar el DataFrame a un archivo CSV."""
    df.to_csv(filename, index=False)
