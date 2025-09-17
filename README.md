# Informe corto sobre el repositorio `low-field-MRI-reconstruction`

## Descripción general

Este proyecto implementa técnicas avanzadas para la reconstrucción de imágenes de resonancia magnética (MRI) adquiridas en equipos de bajo campo. El objetivo principal es mejorar la calidad de las imágenes que suelen ser borrosas o de baja resolución debido a las limitaciones de hardware en ese tipo de equipos.

## ¿Qué hace el proyecto según el código?

- **Procesamiento de imágenes:** Utiliza transformadas wavelet para descomponer imágenes MRI en componentes de aproximación y detalles, permitiendo el análisis y manipulación de la información espacial (archivos como `check.py` y `check2.py`).

- **Reconstrucción automática:** Implementa modelos de redes neuronales profundas (CNN, autoencoders y modelos híbridos como WaCAEN) para restaurar imágenes MRI a partir de versiones degradadas, incrementando la nitidez y la información visual recuperada (`predict_new.py`, `predict_ssim.py`).

- **Evaluación cuantitativa:** Calcula métricas como PSNR, SSIM, MS-SSIM, VIFP, MSE, RMSE, SCC, RASE, SAM para comparar la calidad de la imagen reconstruida frente a la imagen objetivo o referencia. Los resultados se pueden guardar en archivos PDF o tablas Markdown para análisis posterior (`new_statistics.py`).

- **Automatización y experimentación:** El proyecto permite el procesamiento por lotes de imágenes en carpetas específicas, genera informes de comparación y facilita la experimentación eliminando o modificando componentes específicos de la imagen (detalles horizontales, verticales, diagonales).

## Componentes clave

- **Transformada wavelet:** Descomposición y reconstrucción de imágenes para extraer y modificar detalles espaciales.
- **Redes neuronales profundas:** Modelos entrenados para restaurar imágenes de baja calidad.
- **Evaluación con métricas estándar:** PSNR, SSIM, VIFP, entre otros.
- **Generación de informes automáticos:** PDF y CSV para comparar resultados y documentar experimentos.

## Aplicación

Este código es útil para investigadores o profesionales que trabajan con imágenes médicas de bajo campo y buscan maximizar la información diagnóstica mediante técnicas de restauración computacional.


---

# README del repositorio `low-field-MRI-reconstruction`

Este repositorio contiene el código y los recursos para realizar la reconstrucción de imágenes de resonancia magnética (MRI) adquiridas en equipos de bajo campo. El enfoque principal es mejorar la calidad de las imágenes y facilitar su análisis mediante diversas técnicas de procesamiento y evaluación.