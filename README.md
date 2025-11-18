# TP3 – Estimación de tarifas de vuelos domésticos

Proyecto del tercer entregable de Ciencia de Datos: entrenamiento de un modelo de regresión para predecir la tarifa de vuelos domésticos en India y despliegue de un prototipo funcional (API + frontend).

El dataset usado es `dataset/Cleaned_dataset.csv`.

---

## 1. Requisitos previos

- macOS (desarrollado y probado en macOS Sequoia).
- **Python 3.11** instalado (por ejemplo vía Homebrew: `brew install python@3.11`).
- Opcional pero recomendado:
  - Homebrew instalado.
  - Dependencias nativas para librerías de visualización:
    ```bash
    xcode-select --install           # Command Line Tools (si no las tenés)
    brew install pkg-config freetype libpng
    ```

---

## 2. Clonar el repo / ubicarse en el proyecto

Si ya tenés la carpeta, simplemente:

```bash
cd /ruta/a/tu/proyecto/cda
```

Desde aquí deberían verse, entre otros, estos archivos:

- `dataset/Cleaned_dataset.csv`
- `tp3_modelado_vuelos.ipynb`
- `api/main.py`
- `frontend/app.py`
- `requirements.txt`

---

## 3. Crear y activar un entorno virtual (venv)

### Crear el venv

```bash
python3 -m venv .venv
```

### Activar el venv

```bash
source .venv/bin/activate
```

Para salir del entorno virtual más adelante:

```bash
deactivate
```

---

## 4. Instalar dependencias

Con el entorno virtual activado:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Esto instalará, entre otras cosas: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `streamlit`, etc.

> Si aparece algún error relacionado con compilación de `matplotlib`, asegurate de haber ejecutado los pasos de Homebrew y Command Line Tools de la sección de requisitos.

---

## 5. Ejecutar el notebook de modelado

El notebook `tp3_modelado_vuelos.ipynb` entrena varios modelos (Random Forest, XGBoost, MLP), compara resultados y guarda el mejor pipeline ya preprocesado.

### Abrir el notebook

Podés abrirlo con:

- VS Code (extensión Jupyter)  
- Jupyter Lab / Notebook:

  ```bash
  jupyter lab
  ```

### Ejecutar las celdas

1. Asegurate de que el kernel seleccionado use el Python del venv (`.venv`).  
2. Ejecutá todas las celdas, en orden, hasta la sección **“1.13 Guardado del modelo para despliegue”**.  
3. La celda final de esa sección guarda el pipeline en:

   ```text
   artifacts/model_pipeline.joblib
   ```

Si todo salió bien, al finalizar deberías ver el archivo `artifacts/model_pipeline.joblib` creado/actualizado.

---

## 6. Levantar el frontend con Streamlit

Con el entorno virtual activado y el modelo ya guardado en `artifacts/model_pipeline.joblib`:

```bash
streamlit run frontend/app.py
```

Streamlit abrirá la app en el navegador (por defecto `http://localhost:8501`).

La app tiene tres pestañas:

1. **Predicción**  
   - Inputs mínimos: fecha del vuelo, origen, destino, clase.  
   - Usa la aerolínea predominante y duración promedio para esa ruta/clase.  
   - Muestra:
     - Rangos de tarifas estimadas por franjas horarias de salida.  
     - Rangos de tarifas estimadas por franjas de llegada.  
     - Rangos por cantidad de escalas.  
     - Curva de tarifa estimada vs. `Days_left`.  

2. **What-if**  
   - Inputs mínimos: fecha base, origen, destino, clase.  
   - Usa promedios históricos para aerolínea y duración.  
   - Genera varios escenarios (base, cambiar días de anticipación, cambiar escalas, distintas franjas de salida/llegada) y los compara en una tabla.

3. **Análisis del dataset**  
   - Histogramas de `Fare` y `Days_left`.  
   - Top rutas y aerolíneas por cantidad de vuelos.  
   - Gráfico de dispersión `Duration_in_hours` vs `Fare` coloreado por escalas.

> Importante: la app asume que la API está corriendo en `http://localhost:8000`. Podés cambiar la URL base desde la barra lateral de Streamlit si fuera necesario.
> En la versión actual, la app carga el modelo directamente desde `artifacts/model_pipeline.joblib`, sin necesidad de levantar una API.

---

## 8. Flujo completo resumido

1. Clonar/abrir el proyecto y ubicarse en la carpeta raíz.  
2. Crear y activar el entorno virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instalar dependencias:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
4. Abrir y ejecutar el notebook `tp3_modelado_vuelos.ipynb` hasta guardar `artifacts/model_pipeline.joblib`.  
5. Levantar el frontend:
   ```bash
   streamlit run frontend/app.py
   ```
6. Probar la app en el navegador, explorando las pestañas de Predicción, What-if y Análisis del dataset.

Con estos pasos, cualquier persona debería poder reproducir el entrenamiento, validar el modelo y usar el prototipo completo en su máquina local.
