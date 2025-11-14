# Proceso de Migración

## Overview

Este documento resume, bloque por bloque, cómo se trasladó el trabajo del notebook `exploration.ipynb` al código productivo.

### Helper estático para `get_min_diff`

Para reutilizar el mismo cálculo tanto en entrenamiento como en serving se define el helper get_min_diff como un `@staticmethod` dentro del modelo:

```python
@staticmethod
def _get_min_diff(data_row):
    fecha_o = datetime.strptime(data_row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data_row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    return ((fecha_o - fecha_i).total_seconds()) / 60
```
## `model.py`: propósito y migración desde `exploration.ipynb`

### Objetivo del módulo
- Centralizar el pipeline de *feature engineering*, entrenamiento e inferencia en una clase reusable (`DelayModel`) desacoplada del API.
- Se definen los metodos (`preprocess`, `fit`, `predict`).
- Se migra one-hot encoding y selección de 10 features clave.

### Pasos de la migración
1. **Corrección de tipos:** se reemplazó `Union()` por `Union[]` para satisfacer los type-hints de Python 3.12.
2. **Preprocesamiento alineado al notebook:**  
   - `_get_min_diff` replica el cálculo de minutos entre `Fecha-O` y `Fecha-I`.  
   - `preprocess` opera sobre una copia del DataFrame, genera dummies de `OPERA`, `TIPOVUELO`, `MES` y garantiza las 10 columnas top (rellena con 0 si faltan).  
   - El target es genérico: si `target_column` existe se devuelve si no existe se deriva (solo cuando hay timestamps) respetando el nombre solicitado.
3. **Entrenamiento balanceado (`fit`):**  
   - Se utiliza `LogisticRegression(class_weight="balanced")`, replicando la recomendación del notebook (top 10 features + balanceo) sin cálculos manuales para evitar errores.
4. **Inferencia segura (`predict`):**  
   - `predict` exige haber llamado `fit` previamente y arroja un `ValueError` descriptivo si no hay modelo entrenado.
5. **Pruebas de modelo:**  
   - `test_model_predict` se actualizó para reflejar el flujo real (`preprocess -> fit -> predict`).  

### Selección del modelo

Durante la migración se revisaron los dos candidatos finales del notebook (XGBoost y Logistic Regression) con top 10 features y balanceo. Las métricas fueron equivalentes, por lo que se priorizó Logistic Regression por motivos operativos:

1. **Dependencias livianas:** 
   - Viene en scikit-learn.
   - XGBoost es necesario agregar una librería.
2. **Consumo de recursos:** 
   - LR entrena e infiere en segundos (celdas 53 tarda 0.2 segundos). 
   - XGBoost tarda más y consume más CPU/RAM (celdas 47 tarda 0,7 segndos), algo crítico para el ambiente donde se va a realizar el deploy (e.g. Cloud Run).

Con métricas similares, se adoptó Logistic Regression con `class_weight="balanced"` para minimizar complejidad operativa sin sacrificar calidad.

## Cloud Environment

1. **Proveedor**: se eligió GCP usando Cloud Run por su naturaleza serverless y porque expone directamente un endpoint público.
2. **Seguridad**: se configuró Workload Identity Federation entre GitHub Actions y GCP para autenticarse sin descargar claves JSON de la Service Account.
3. **Artefactos**: las imágenes Docker se almacenan en Artifact Registry, para en despliegues manuales como en CI/CD.

## `api.py`: exposición del modelo
1. **Logging definido para mejor manejo de errores:** se agrega el import de logging para poder manejar los erroes.
2. **Carga determinística de datos:** `DelayService` ahora lee `data.csv` mediante un helper que fija `dtype` para `Vlo-I` y `Vlo-O` y usa `low_memory=False`.
3. **Entrenamiento con logging:** `ensure_model` registra el resultado del entrenamiento y propaga las excepciones para que el `lifespan`/endpoint puedan responder con errores claros.
4. **Validaciones de request:** `FlightRequest` valida que la lista de vuelos no venga vacía, en conjunto con las validaciones previas de `OPERA`, `TIPOVUELO` y `MES`.
5. **Manejo de errores de inferencia:** `/predict` mantiene las respuestas HTTP 400 para problemas de validación y 500 para fallas inesperadas, respaldado por logging que facilita el troubleshooting.

## Dockerfile y despliegue en GCP

1. **Dockerfile**  
   - Basado en `python:3.12-slim`, solamente instala `requirements.txt`, copia el paquete `challenge` y expone Uvicorn en `0.0.0.0:8080`.
2. **Cloud Run**  
   - Servicio productivo: `challenge-api` en región configurada.  
   - La URL resultante se replica en el `Makefile` para `make stress-test`.g
3. **Seguridad**  
   - Se usa Artifact Registry como repositorio de imágenes y Workload Identity Federation (pool `github-pool`, provider `github-provider`) para evitar claves JSON en CD.

