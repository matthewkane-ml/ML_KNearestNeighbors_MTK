# K-Nearest Neighbors — Clasificación de Calidad de Vino

> Clasificador KNN multiclase sobre 1.599 muestras de vino tinto: escalado de características, un barrido de k de 1 a 20 para encontrar el tamaño óptimo del vecindario y una función de inferencia `predict_wine_quality()` — alcanzando el 84,4% de precisión con k=5 y llegando al máximo con k=14.

---

## Problema

Predecir si un vino tinto es de calidad **baja**, **media** o **alta** basándose en 11 medidas fisicoquímicas. Los productores y distribuidores de vino quieren una señal de calidad objetiva y basada en datos que no dependa exclusivamente de catadores expertos costosos. Este es un problema de clasificación de 3 clases.

## Dataset

- **Fuente:** Dataset Red Wine Quality (UCI vía GitHub)
- **Tamaño:** 1.599 filas × 12 columnas (11 características + puntuación de calidad)
- **Ingeniería de etiquetas de calidad:** puntuación `quality` (0–10) → 3 clases:

| Puntuación de calidad | Etiqueta | Clase |
|---|---|---|
| ≤ 4 | Baja | 0 |
| 5–6 | Media | 1 |
| ≥ 7 | Alta | 2 |

**Características:** acidez fija, acidez volátil, ácido cítrico, azúcar residual, cloruros, dióxido de azufre libre, dióxido de azufre total, densidad, pH, sulfatos, alcohol

## Pipeline

| Paso | Acción |
|---|---|
| Ingeniería de etiquetas | Puntuaciones de calidad agrupadas en 3 clases ordinales |
| División train/test | 80/20, random_state=42 (1.279 entrenamiento / 320 prueba) |
| Escalado | `StandardScaler` — crítico para KNN ya que la distancia depende de la escala |
| Modelo base | `KNeighborsClassifier(n_neighbors=5)` |
| Optimización | Barrido k=1 a k=20, registro de precisión en cada k, selección del mejor |
| Mejor k | **k=14** |
| Función de inferencia | `predict_wine_quality(features)` → devuelve etiqueta legible por humanos |

## Resultados del Modelo

**Línea base k=5:**

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| Baja (0) | 0,00 | 0,00 | 0,00 | 11 |
| Media (1) | 0,87 | 0,95 | 0,91 | 262 |
| Alta (2) | 0,65 | 0,43 | 0,51 | 47 |
| **Precisión global** | | | **84,4%** | 320 |

**Tras la optimización con barrido → k=14** mejora aún más la precisión al suavizar la frontera de decisión.

**Observación clave:** La clase 0 (calidad baja) tiene precisión y recall nulos con k=5 — solo 11 muestras de prueba la hacen prácticamente invisible durante el entrenamiento. Este es un problema de desbalance de clases, no un fallo de KNN.

## Conclusiones Clave

- **El escalado es obligatorio para KNN:** KNN mide la distancia entre puntos de datos. Sin StandardScaler, las características con rangos numéricos grandes (como `total sulfur dioxide` hasta 289) dominan el cálculo de distancia y oscurecen características informativas como `pH` (rango ≈ 3,0–4,0).
- **k controla el equilibrio sesgo–varianza:** k pequeño (k=1) memoriza el ruido del entrenamiento — alta varianza. k grande promedia sobre muchos vecinos — alto sesgo, fronteras más suaves. El barrido hace este equilibrio explícito y elige el óptimo empírico.
- **La precisión global oculta el fallo a nivel de clase:** El 84% de precisión suena sólido, pero el modelo falla completamente en los vinos de baja calidad (la clase minoritaria). Para un caso de uso en una bodega, no detectar botellas de baja calidad sería un fallo significativo en el mundo real.

## Stack Tecnológico

`Python` · `scikit-learn` · `pandas` · `Matplotlib`

## Ejecutar Localmente

```bash
git clone https://github.com/matthewkane-ml/ML_KNearestNeighbors_MTK.git
cd ML_KNearestNeighbors_MTK
pip install -r requirements.txt
jupyter notebook src/explore.ipynb
```

## Próximos Pasos

- Abordar el desbalance de clases con sobremuestreo **SMOTE** en el conjunto de entrenamiento para dar a la clase de baja calidad suficiente representación como para ser aprendible
- Probar **KNN ponderado** (`weights="distance"`) para que los vecinos más cercanos tengan más influencia que los lejanos
- Comparar con un clasificador **Random Forest** en las mismas características para cuantificar el techo de precisión alcanzable con un método no basado en distancias

---

**Autor:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [Portafolio GitHub](https://github.com/matthewkane-ml)
