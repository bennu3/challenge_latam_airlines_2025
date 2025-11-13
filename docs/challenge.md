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
1. **Se corrige bug en la llamada de Union():** Se cambia los `()` por `[]` apra corregir el error de `Call expression not allowed in type expression`
2. **Extracción del preprocesamiento:**  
   - Se implementó el helper `_get_min_diff` para replicar el cálculo de la diferencia de minutos entre `Fecha-O` y `Fecha-I`.  
   - `preprocess` aplica `pd.get_dummies` sobre `OPERA`, `TIPOVUELO` y `MES` y garantiza que las 10 columnas más relevantes (de no existir se asigna el valor cero).  
3. **Entrenamiento reproducible (`fit`):**  
   - Se entrena una única instancia de `LogisticRegression`, almacenada en `self._model`, dejando el modelo listo para ser consumido por la API.
4. **Inferencia segura (`predict`):**  
   - Antes de predecir se verifica que el modelo haya sido entrenado. En caso contrario se reutiliza el último batch cacheado para ejecutar `fit` de manera lazzy.  
   - Las predicciones se devuelven como lista de enteros, respetando el contrato del endpoint `/predict`.
