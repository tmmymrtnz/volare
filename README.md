# TP3 – Estimación de tarifas de vuelos domésticos

Proyecto del tercer entregable de Ciencia de Datos: entrenamiento de un modelo de regresión para predecir la tarifa de vuelos domésticos en India y despliegue de un prototipo funcional (API + frontend).

El dataset usado es `dataset/Cleaned_dataset.csv`. **Nota**: Si el archivo no existe localmente, el notebook lo descargará automáticamente desde Google Drive al ejecutarse.

---

## 1. Requisitos previos

- macOS (desarrollado y probado en macOS Sequoia).
- **Python 3.11** instalado (por ejemplo vía Homebrew: `brew install python@3.11`).
- Opcional pero recomendado:
  - Homebrew instalado.
  - Dependencias nativas para librerías de visualización y cómputo paralelizado:
    ```bash
    xcode-select --install           # Command Line Tools (si no las tenés)
    brew install pkg-config freetype libpng libomp
    ```

---

## 2. Clonar el repo / ubicarse en el proyecto

Si ya tenés la carpeta, simplemente:

```bash
cd /ruta/a/tu/proyecto/cda
```

Desde aquí deberían verse, entre otros, estos archivos:

- `dataset/` (el archivo `Cleaned_dataset.csv` se descargará automáticamente si no existe)
- `tp3_modelado_vuelos.ipynb`
- `api/main.py`
- `frontend/app.py`
- `volare_model/`
- `requirements.txt`
- `reports/`

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

Esto instalará, entre otras cosas: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `fastapi`, `uvicorn`, `streamlit`, `requests`, etc.

> Si aparece algún error relacionado con compilación de `matplotlib`, asegurate de haber ejecutado los pasos de Homebrew y Command Line Tools de la sección de requisitos.

### Verificar la instalación

Para confirmar que todo se instaló correctamente:

```bash
python -c "import pandas, sklearn, xgboost, streamlit, fastapi; print('✓ Dependencias principales instaladas')"
```

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
2. **Primera ejecución**: Si el archivo `dataset/Cleaned_dataset.csv` no existe, la celda de carga lo descargará automáticamente desde Google Drive (esto puede tomar unos minutos).
3. Ejecutá todas las celdas, en orden, hasta la sección **"1.14 Guardado del modelo para despliegue"**.  
4. La celda final de esa sección guarda el pipeline y las métricas consolidadas en:

   ```text
   artifacts/model_pipeline.joblib
   reports/model_metrics.json
   ```

El notebook ahora:

- Ordena el dataset cronológicamente y usa `TimeSeriesSplit` para evaluar sin leakage.
- Ajusta hiperparámetros con `RandomizedSearchCV` y genera curvas de aprendizaje.
- Serializa el mejor pipeline (`FeatureGenerator` + `XGBoost` tuneado) y exporta sus métricas.
- Exporta diagnósticos (bootstrap, MAE por buckets, stress tests) y centraliza métricas en `reports/model_metrics.json`.

Si todo salió bien, al finalizar deberías ver ambos artefactos actualizados (`artifacts/` y `reports/`).

---

## 6. Levantar el frontend con Streamlit

Con el entorno virtual activado y el modelo ya guardado en `artifacts/model_pipeline.joblib`:

```bash
streamlit run frontend/app.py
```

Streamlit abrirá la app en el navegador (por defecto `http://localhost:8501`). Desde la barra lateral podés elegir el modo de inferencia:

- **Pipeline local**: llama al modelo cargado desde `artifacts/model_pipeline.joblib`.
- **API FastAPI**: envía los payloads a `POST /predict` (URL configurable).

La app incluye cinco pestañas:

1. **Predicción** – calcula rangos por horarios de salida/llegada, escalas, aerolínea y curva vs `Days_left`.  
2. **What-if** – escenarios comparativos (cambiar anticipación, escalas, franjas).  
3. **Casos típicos** – presets para viajes de negocios, planificados o escapadas.  
4. **Predicción completa** – formulario avanzado con todos los parámetros.  
5. **Análisis del dataset** – histogramas, top rutas/aerolíneas y scatter `Duration_in_hours` vs `Fare`.

> Si elegís “API FastAPI”, asegurate de tener `uvicorn api.main:app --reload` corriendo (ver sección 7). En modo local, no se requiere la API.

---

## 7. Servir la API FastAPI

Con el entorno virtual activo y el modelo generado:

```bash
uvicorn api.main:app --reload --port 8000
```

Endpoints principales:

- `GET /health` → chequea estado y versión del artefacto cargado.
- `POST /predict` → recibe el mismo payload que usa la app (campos como `Date_of_journey`, `Airline`, `Total_stops`, etc.) y devuelve la tarifa estimada + metadata.

Ejemplo rápido:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "Date_of_journey": "2024-07-01",
    "Journey_day": "Monday",
    "Airline": "Indigo",
    "Class": "Economy",
    "Source": "Delhi",
    "Departure": "6 AM - 12 PM",
    "Total_stops": "non-stop",
    "Arrival": "12 PM - 6 PM",
    "Destination": "Mumbai",
    "Duration_in_hours": 2.3,
    "Days_left": 15
  }'
```

La API comparte el mismo `LocalModelService` que la app, por lo que cualquier artefacto nuevo generado por el notebook puede desplegarse sin cambios adicionales.

---

## 8. Propuesta de despliegue

| Componente | Responsabilidad | Comando base | Hosting sugerido |
| --- | --- | --- | --- |
| Notebook (`tp3_modelado_vuelos.ipynb`) | Entrenamiento batch, exporta `artifacts/` y `reports/` | `jupyter lab` / `papermill` | EC2 puntual, Workbench o GitHub Actions + S3 |
| API (`api/main.py`) | Inferencia online (`POST /predict`) | `uvicorn api.main:app --port 8000` | Contenedor (ECS/K8s) detrás de ALB/API Gateway |
| Frontend (`frontend/app.py`) | Experiencia interactiva para analistas/usuarios finales | `streamlit run frontend/app.py` | Streamlit Cloud, EC2 con Nginx o S3+CloudFront (via Streamlit Sharing) |
| Artefactos (`artifacts/`, `reports/`) | Versionado del modelo y métricas | n/a | S3 / GCS + control de versiones |

Escalabilidad sugerida:

- **Inferencia**: autoscaling horizontal + caché en Redis para rutas populares. Cada pod carga el `LocalModelService`.
- **Entrenamiento**: job mensual/semanal que ejecuta el notebook (o script equivalente) y publica métricas; sólo promueve nuevas versiones si superan el MAE anterior.
- **Observabilidad**: registrar MAE por ruta y buckets de `Days_left`, alertar si se sale del intervalo bootstrap calculado en el notebook.

---

## 9. Flujo completo resumido

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
4. Abrir y ejecutar el notebook `tp3_modelado_vuelos.ipynb` hasta guardar `artifacts/model_pipeline.joblib` + `reports/model_metrics.json`.  
5. (Opcional) Levantar la API:
   ```bash
   uvicorn api.main:app --reload
   ```
6. Levantar el frontend:
   ```bash
   streamlit run frontend/app.py
   ```
7. Probar la app en el navegador, explorando las pestañas de Predicción, What-if, Casos, Predicción completa y Análisis.

Con estos pasos, cualquier persona debería poder reproducir el entrenamiento, validar el modelo y usar el prototipo completo en su máquina local.

---

## 10. Troubleshooting

### El notebook no encuentra el dataset

- El notebook descarga automáticamente el dataset desde Google Drive si no existe. Si falla la descarga:
  - Verificá tu conexión a internet.
  - Asegurate de que la carpeta `dataset/` exista (se crea automáticamente).
  - Si el problema persiste, podés descargar manualmente el archivo desde Google Drive usando el ID: `183MypWCEwXVmQyawp6U0DZpcWs8AEsIJ`.

### Error al importar módulos en el notebook

- Verificá que el kernel de Jupyter esté usando el entorno virtual `.venv`.
- En VS Code: seleccioná el kernel desde la esquina superior derecha del notebook.
- En Jupyter Lab: `Kernel` → `Change Kernel` → seleccioná el entorno `.venv`.

### La API o el frontend no encuentran el modelo

- Asegurate de haber ejecutado el notebook completo hasta la sección "1.14 Guardado del modelo para despliegue".
- Verificá que exista el archivo `artifacts/model_pipeline.joblib`.
- Si el archivo existe pero hay errores, probá regenerarlo ejecutando nuevamente la celda de guardado del modelo.

### Error de puerto en uso

- Si el puerto 8000 (API) o 8501 (Streamlit) está ocupado:
  - Para la API: `uvicorn api.main:app --reload --port 8001`
  - Para Streamlit: `streamlit run frontend/app.py --server.port 8502`
