Analsis de sentimientos basado en Tweets:
Este proyecto tiene como objetivo realizar un análisis de sentimientos de tweets relacionados con COVID-19 utilizando la biblioteca VADER de NLTK. El análisis de sentimientos clasifica los tweets en categorías de 'Positivo', 'Negativo' y 'Neutral' y visualiza los resultados en gráficos circulares.

Requisitos: 
- Google colab
- Google Drive
- Pandas
- NLTK
- Matplotlib
Instrucciones:
1.Montar Google Drive en Google Colab
Primero, montamos Google Drive en Google Colab para acceder a los archivos almacenados allí.
```
from google.colab import drive
drive.mount('/content/drive')
```
2. Importar la Biblioteca Pandas
Pandas se utiliza para manejar y manipular datos.
```
import pandas as pd
```
3. Leer el Archivo CSV desde Google Drive
Leemos el archivo CSV que contiene los tweets sobre COVID-19 o cualquier otra BD de tengamos, para el proyecto se adjuntaron 2 BD, una sobre tusimo y otra sobre el COVID-19.
```
file_path = "/content/drive/MyDrive/covid19_tweets.csv"
datos = pd.read_csv(file_path)
# Mostrar los datos para verificar que se hayan cargado correctamente
print(datos)
```
4. Instalar y Descargar NLTK
NLTK (Natural Language Toolkit) es una biblioteca de procesamiento de lenguaje natural.
```
!pip install nltk
import nltk
nltk.download('vader_lexicon')
```
5. Importar las Bibliotecas Necesarias
Importamos el analizador de sentimientos VADER de NLTK.
```
from nltk.sentiment.vader import SentimentIntensityAnalyzer
```
6. Inicializar el Analizador de Sentimientos VADER
Inicializamos el analizador de sentimientos.
```
sid = SentimentIntensityAnalyzer()
```
7. Definir la Función para Asignar Etiquetas de Sentimiento
Creamos una función para asignar una etiqueta de sentimiento a cada tweet basado en la puntuación compuesta.
```
def get_sentimientos(tweet):
    scores = sid.polarity_scores(tweet)
    if scores['compound'] > 0.05:
        return 'Positivo'
    elif scores['compound'] < -0.05:
        return 'Negativo'
    else:
        return 'Neutral'
```
8. Aplicar la Función a los Tweets
Aplicamos la función a cada tweet usando el campo de la BD asigando para dicho Tweet, el nombre del campo puede variar en cada BD, seguido creamos una nueva columna llamada "Sentimiento".
```
datos['Sentimiento'] = datos['text'].apply(get_sentimientos)
# Mostrar los primeros tweets con sus etiquetas de sentimiento
print(datos[['text', 'Sentimiento']])
```
9. Visualizar los Resultados
Visualizamos los resultados en un gráfico circular.
```
import matplotlib.pyplot as plt
# Contar la cantidad de tweets en cada categoría de sentimiento
sentimiento_counts = datos['Sentimiento'].value_counts()

# Calcular el porcentaje de tweets en cada categoría de sentimiento
total_tweets = len(datos)
porcentaje_positivo = (sentimiento_counts['Positivo'] / total_tweets) * 100
porcentaje_negativo = (sentimiento_counts['Negativo'] / total_tweets) * 100
porcentaje_neutral = (sentimiento_counts['Neutral'] / total_tweets) * 100

# Etiquetas y porcentajes para el gráfico circular
etiquetas = ['Positivo', 'Negativo', 'Neutral']
porcentajes = [porcentaje_positivo, porcentaje_negativo, porcentaje_neutral]

# Crear el gráfico circular
plt.figure(figsize=(8, 6))
plt.pie(porcentajes, labels=etiquetas, autopct='%1.1f%%', startangle=140)
plt.title('Porcentaje de Tweets por Sentimiento')
plt.axis('equal')
plt.show()
```
10. Agregar un Nuevo Tweet
También podemos agregar un nuevo Tweet manualmente donde puede o no variar el resultado del diagrama dependiendo su tamaño.
```
# Agregar el nuevo tweet al conjunto de datos
nuevo_tweet = "Este es un nuevo tweet sobre COVID-19. #COVID19 #nuevo"
nuevo_tweet_sentimiento = get_sentimientos(nuevo_tweet)
# Crear un nuevo DataFrame con el nuevo tweet
nuevo_tweet_df = pd.DataFrame({'text': [nuevo_tweet], 'Sentimiento': [nuevo_tweet_sentimiento]})
# Concatenar el nuevo DataFrame con el conjunto de datos existente
datos = pd.concat([datos, nuevo_tweet_df], ignore_index=True)
# Recalcular los porcentajes
sentimiento_counts = datos['Sentimiento'].value_counts()
total_tweets = len(datos)
porcentaje_positivo = (sentimiento_counts['Positivo'] / total_tweets) * 100
porcentaje_negativo = (sentimiento_counts['Negativo'] / total_tweets) * 100
porcentaje_neutral = (sentimiento_counts['Neutral'] / total_tweets) * 100
etiquetas = ['Positivo', 'Negativo', 'Neutral']
porcentajes = [porcentaje_positivo, porcentaje_negativo, porcentaje_neutral]
# Crear el gráfico circular actualizado
plt.figure(figsize=(8, 6))
plt.pie(porcentajes, labels=etiquetas, autopct='%1.1f%%', startangle=140)
plt.title('Porcentaje de Tweets por Sentimiento')
plt.axis('equal')
plt.show()
```
Conclusión:
Este proyecto demuestra cómo se puede utilizar la biblioteca VADER de NLTK para realizar un análisis de sentimientos de tweets. Los resultados se visualizan en gráficos circulares, lo que permite una interpretación rápida y efectiva de la distribución de sentimientos en los tweets analizados.
