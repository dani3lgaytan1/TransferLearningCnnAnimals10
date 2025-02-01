# Modelo de Clasificación de Imágenes Usando Redes Neuronales Convolucionales

Este repositorio contiene el código que construye y compila una Red Neuronal Convolucional (CNN) utilizando TensorFlow y Keras para la clasificación de imágenes del dataset Animals10. El modelo está diseñado para clasificar imágenes de 10 categorías y utiliza Global Average Pooling para reducir el número de parámetros y ayudar a prevenir el sobreajuste.

---

## Índice

- [Resumen](#resumen)
- [Arquitectura](#arquitectura)
- [Tecnologías](#tecnologías)
- [Instalación](#instalación)
- [Uso](#uso)
- [Explicación del Código](#explicación-del-código)
- [Licencia](#licencia)

---

## Resumen

El proyecto implementa una arquitectura de CNN con las siguientes características clave:

- **Múltiples Capas Convolucionales:** Se utilizan tres capas convolucionales con un número creciente de filtros (32, 64 y 128) para capturar características en diferentes niveles de abstracción.
- **Capas de Pooling:** Cada capa convolucional es seguida por una capa MaxPooling2D con un tamaño de pool de 2x2 y padding "same", lo que reduce las dimensiones espaciales manteniendo las características importantes.
- **Global Average Pooling:** En lugar de usar una capa Flatten, se aplica GlobalAveragePooling2D para calcular el promedio de cada mapa de características, reduciendo significativamente el número de parámetros y ayudando a combatir el sobreajuste.
- **Capas Densas con Regularización:** Se añade una capa densa con 64 neuronas para una mayor extracción de características, seguida por una capa Dropout (con una tasa de 30%) para mejorar la generalización. La capa densa final utiliza activación softmax para la clasificación multiclase.
- **Compilación:** El modelo se compila utilizando el optimizador Adam y la función de pérdida *categorical_crossentropy*, evaluándose con la métrica de precisión.

---

## Arquitectura

La red está estructurada de la siguiente manera:

1. **Capa de Entrada:**
   - El modelo espera una forma de entrada basada en los datos de entrenamiento (por ejemplo, altura, anchura y número de canales).

2. **Bloque Convolucional 1:**
   - **Conv2D:** 32 filtros, kernel de 3x3, activación ReLU.
   - **MaxPooling2D:** Pooling de 2x2, padding "same".

3. **Bloque Convolucional 2:**
   - **Conv2D:** 64 filtros, kernel de 3x3, activación ReLU.
   - **MaxPooling2D:** Pooling de 2x2, padding "same".

4. **Bloque Convolucional 3:**
   - **Conv2D:** 128 filtros, kernel de 3x3, activación ReLU.
   - **MaxPooling2D:** Pooling de 2x2, padding "same".

5. **Global Average Pooling:**
   - **GlobalAveragePooling2D:** Reduce cada mapa de características a un solo número mediante el promedio, lo que disminuye el número total de parámetros.

6. **Capas Completamente Conectadas:**
   - **Dense:** 64 neuronas con activación ReLU.
   - **Dropout:** Tasa de 30% para regularización.
   - **Dense (Salida):** 5 neuronas con activación softmax para la clasificación.

7. **Compilación:**
   - Función de pérdida: Categorical Cross-Entropy.
   - Optimizador: Adam.
   - Métrica: Precisión.

---

## Tecnologías

- **TensorFlow & Keras:** Para construir y entrenar la CNN.
- **Python:** Lenguaje de programación utilizado para la implementación.

---



## Explicación del Código

A continuación se resume lo que hace el código:

1. **Importación de Capas e Inicialización del Modelo:**

   ```python
   from tensorflow.keras.layers import GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dense, Dropout
   from tensorflow.keras.models import Sequential

   model = Sequential()
   ```

2. **Añadiendo Capas Convolucionales y de Pooling:**

   - **Primer Bloque Convolucional:**
     ```python
     model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation="relu"))
     model.add(MaxPooling2D(2, 2, padding="same"))
     ```
   - **Segundo Bloque Convolucional:**
     ```python
     model.add(Conv2D(64, (3, 3), activation="relu"))
     model.add(MaxPooling2D(2, 2, padding="same"))
     ```
   - **Tercer Bloque Convolucional:**
     ```python
     model.add(Conv2D(128, (3, 3), activation="relu"))
     model.add(MaxPooling2D(2, 2, padding="same"))
     ```

3. **Global Average Pooling:**

   ```python
   model.add(GlobalAveragePooling2D())
   ```

   - Esta capa calcula el promedio de salida de cada mapa de características, reduciendo los datos de un tensor multidimensional a un vector por muestra.

4. **Capas Densas y Dropout:**

   ```python
   model.add(Dense(64, activation="relu"))
   model.add(Dropout(0.3))
   model.add(Dense(5, activation="softmax"))
   ```

   - Se añade una capa completamente conectada con 64 neuronas y activación ReLU.
   - Se aplica una capa Dropout para prevenir el sobreajuste.
   - La capa final usa activación softmax para generar las probabilidades de cada clase.

5. **Compilación del Modelo:**

   ```python
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
   ```

   - El modelo se compila con la función de pérdida categorical cross-entropy, el optimizador Adam y la métrica de precisión.

6. **Resumen del Modelo:**

   ```python
   model.summary()
   ```

   - Se muestra un resumen de la arquitectura del modelo, incluyendo el número de parámetros y la forma de salida de cada capa.

---
