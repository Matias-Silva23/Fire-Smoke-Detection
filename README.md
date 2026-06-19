# Sistema de Monitoreo y Deteccion Temprana de Incendios Forestales con PyTorch Mobile

Este repositorio contiene la implementacion, el analisis de rendimiento y el despliegue de un modelo de Procesamiendo Digital de Imagenes de  basado en la arquitectura YOLOv8 Large. El sistema ha sido optimizado mediante la compilacion en TorchScript para garantizar su portabilidad y ejecucion eficiente en dispositivos con recursos computacionales limitados, encontrandose desplegado interactivamente en Hugging Face Spaces.

---

## 1. Documentacion Obligatoria de la Base de Datos

Para el entrenamiento y validacion de este modelo se utilizo la version optimizada y reestructurada del dataset D-Fire (Dataset for Smoke and Fire Detection).

### Ficha Tecnica del Dataset
* Nombre Oficial: D-Fire Dataset (Enhanced Version)
* Creadores Originales: Investigadores de GAIA (Solutions on Demand).
* Proposito: Facilitar tareas de deteccion e identificacion de amenazas criticas mediante aprendizaje profundo en entornos silvestres y urbanos.
* Formato de Anotacion: Cajas delimitadoras (Bounding Boxes) estandarizadas en el formato YOLO (archivos .txt normalizados).

### Estructura del Directorio
El conjunto de datos se organizo de forma estrictamente particionada para garantizar la evaluacion del aprendizaje:

```text
train/
  ├── images/      # Imagenes de entrenamiento
  └── labels/      # Etiquetas en formato YOLO
val/
  ├── images/      # Imagenes de validacion
  └── labels/      # Etiquetas en formato YOLO
test/
  ├── images/      # Imagenes de prueba
  └── labels/      # Etiquetas en formato YOLO

# Estadísticas y Clases del Dataset

El dataset incluye anotaciones específicas para las siguientes clases identificadas por sus índices numéricos:

- **Clase 0:** Fuego (*Fire*)
- **Clase 1:** Humo (*Smoke*)

El conjunto de datos comprende un volumen robusto de **12,431 imágenes**, categorizadas en función de su contenido de la siguiente manera:

- **Imágenes con presencia de fuego:** 8,554 imágenes.
- **Imágenes con presencia de humo:** 3,609 imágenes.
- **Imágenes con presencia simultánea de fuego y humo:** 1,732 imágenes.
- **Imágenes de control sin elementos de riesgo (fondos limpios):** 2,000 imágenes.

Respecto al volumen de objetos etiquetados, el dataset cuenta con el siguiente total de cajas delimitadoras (*Bounding Boxes*):

- **Total de cajas delimitadoras para la clase Fuego:** 14,692 anotaciones.
- **Total de cajas delimitadoras para la clase Humo:** 11,865 anotaciones.

# 2. Métricas de Rendimiento Obtenidas (Época 80)

El entrenamiento se completó exitosamente en **80 épocas** utilizando aceleradores de hardware (*Multi-GPU*). Mediante la evaluación final en el conjunto de validación (compuesto por **1,865 imágenes independientes** que el modelo jamás vio durante el entrenamiento), se obtuvieron resultados de alto rendimiento y confiabilidad.

## Métricas Globales de Validación

- **mAP@50 (Precisión General Promedio):** 0.7775 (77.75%)
- **mAP@50-95 (Criterio Estricto de Traslape):** 0.4823 (48.23%)


## Métricas por Clase

### Clase [0]: Fuego

- **Precisión (Exactitud):** 83.19%
- **Recall (Sensibilidad):** 79.07%

**Descripción técnica:**  
Balance sobresaliente. De cada 100 alertas, aproximadamente 83 corresponden a fuego real (bajo índice de falsos positivos) y detecta el 79% de los focos de incendio en las imágenes.

### Clase [1]: Humo

- **Precisión (Exactitud):** 77.64%
- **Recall (Sensibilidad):** 66.32%

**Descripción técnica:**  
Rendimiento robusto en una clase compleja. El modelo discrimina eficazmente las columnas de humo frente a nubes u otros elementos del paisaje, manteniendo una sensibilidad adecuada ante la naturaleza amorfa del gas.

# 3. Discusión sobre la Generalización y Ausencia de Sobreajuste (Overfitting)

El modelo finalizado en la época 80 demuestra una excelente capacidad de generalización. En el aprendizaje profundo, un modelo sobreajustado (*overfitting*) tiende a memorizar el conjunto de entrenamiento, lo que provoca que su sensibilidad (*Recall*) disminuya drásticamente al enfrentarse a un conjunto de validación nuevo.

En este proyecto, mantener una precisión global cercana al **80%** y un *Recall* del **79.07%** para la clase **Fuego** y del **66.32%** para la clase **Humo** sobre casi **2,000 imágenes de validación independientes** constituye evidencia de que la red neuronal **Large** adquirió un criterio de detección robusto y flexible.

Las curvas de pérdida (*loss*) decrecieron de manera proporcional durante el entrenamiento, indicando una convergencia estable y una adecuada capacidad de generalización. Esto sugiere que el sistema se encuentra preparado para operar de forma confiable en escenarios del mundo real.

# 4. Pipeline de Optimización y Exportación a TorchScript

Con el fin de mitigar problemas de incompatibilidad entre la API de **ExecuTorch** y la versión nativa de **PyTorch** instalada en el entorno de desarrollo, se optó por una estrategia basada en **TorchScript**.

**TorchScript** permite serializar y optimizar modelos de PyTorch para ejecutarlos en entornos independientes de Python, incluyendo aplicaciones en C++, sistemas embebidos y soluciones de inferencia móvil de alto rendimiento.

## Parámetros de Optimización Aplicados

### `optimize=True`

Aplica optimizaciones a nivel de grafo computacional y capas internas del modelo, reduciendo operaciones redundantes y mejorando la eficiencia de ejecución sobre el hardware objetivo.

### `nms=False`

Desactiva la implementación nativa de **Non-Maximum Suppression (NMS)** dentro del grafo exportado. Esta decisión reduce la complejidad computacional y el tamaño del modelo, delegando dicha operación al entorno de inferencia final para maximizar el rendimiento y los cuadros por segundo (**FPS**).

# 5. Despliegue e Inferencia en Hugging Face Spaces

La integración en Hugging Face Spaces utiliza una arquitectura de repositorio estructurada que funciona tanto como interfaz web interactiva para los usuarios como punto centralizado de distribución del modelo optimizado para dispositivos de campo.

## Estructura del Space

### `requirements.txt`

Declara las dependencias necesarias para la ejecución del servicio en la nube, incluyendo:

- `ultralytics`
- `gradio`
- `torch`
- `torchvision`

### `best.torchscript`

Archivo serializado y optimizado generado durante el proceso de exportación. Contiene el modelo listo para inferencia en entornos compatibles con TorchScript.

### `app.py`

Archivo principal de la aplicación. Implementa la interfaz gráfica de usuario mediante la biblioteca **Gradio**, gestionando la carga de imágenes, la ejecución de inferencias y la visualización de resultados.

## Mecanismo de Inferencia en el Servidor

Durante la inicialización del servicio, el entorno de ejecución carga el archivo `best.torchscript` y lo conecta con la interfaz desarrollada en Gradio.

Cuando un operador carga una imagen, esta es procesada por el modelo optimizado, el cual ejecuta el proceso de detección sobre el contenido visual y retorna las coordenadas de las regiones identificadas junto con sus respectivas clases y niveles de confianza.

# 6. Valor Arquitectónico del Proyecto

La adopción de **TorchScript** como formato de despliegue frente a los desafíos de compatibilidad entre bibliotecas es una estrategia de resolución adaptativa de problemas dentro de un flujo de trabajo.

Al compilar el modelo bajo este formato, se elimina la dependencia directa de la ejecución dinámica del código fuente en Python, permitiendo que el modelo sea distribuido como un artefacto autocontenido y optimizado para inferencia.

Esta decisión arquitectónica ofrece múltiples ventajas:

- Mayor portabilidad entre entornos de ejecución.
- Menor dependencia de versiones específicas de bibliotecas.
- Facilidad de integración en aplicaciones nativas.
- Reducción de la complejidad de despliegue.
- Posibilidad de ejecución en dispositivos con recursos limitados.

El archivo resultante puede ser integrado directamente en aplicaciones desarrolladas en C++, sistemas embebidos, plataformas móviles o soluciones de monitoreo autónomas, alineándose con el objetivo de construir un sistema de detección de incendios de baja latencia y alta disponibilidad para escenarios rurales y ambientes con conectividad limitada.

La arquitectura final demuestra que el modelo no solo alcanza un desempeño competitivo en términos de detección, sino que además cumple con requisitos prácticos de despliegue para aplicaciones reales de monitoreo ambiental.

# 7. Citación y Créditos Originales

Este proyecto se desarrolló utilizando como base el conjunto de datos **D-Fire**,