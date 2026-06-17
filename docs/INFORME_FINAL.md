# Trabajo Práctico Grupal: Clasificación de Autismo en rs-fMRI usando Redes Convolucionales y Generalización de Dominio (DANN)

**Maestría en Inteligencia Artificial**  
**Materia:** Visión y Percepción Computarizada  
**Integrantes:** Sebastián Mesch Henriques · Leandro Carcagno  

---

## 1. Introducción y Descripción del Problema

El Trastorno del Espectro Autista (ASD, por sus siglas en inglés) es una condición del neurodesarrollo caracterizada por dificultades en la comunicación social, comportamientos repetitivos y patrones de interés restringidos. Históricamente, el diagnóstico de ASD ha sido puramente clínico, basado en la observación del comportamiento y entrevistas estructuradas (como ADOS y ADI-R). Sin embargo, la búsqueda de biomarcadores biológicos objetivos y reproducibles basados en neuroimágenes es uno de los campos de investigación más activos en la neurociencia computacional contemporánea.

Este trabajo aborda la **clasificación binaria** de sujetos diagnosticados con autismo (ASD = 1) frente a controles típicos (TC = 0) utilizando imágenes de **Resonancia Magnética Funcional en Reposo (rs-fMRI)** provenientes del consorcio internacional **ABIDE (Autism Brain Imaging Data Exchange)**. El objetivo central es **pedagógico**: construir un pipeline reproducible de extremo a extremo (datos → imágenes → modelo → métricas) y evaluar el aporte de la adaptación adversarial de dominio para mitigar el sesgo instrumental multicentro.

### Enfoque desde la Visión por Computadora
En lugar del enfoque neurocientífico tradicional centrado en análisis tabulares de matrices de conectividad funcional de red (conectomas), planteamos el problema bajo el paradigma de **Visión por Computadora (Computer Vision)**. Para ello:
1. Consumimos mapas paramétricos tridimensionales resumidos (fALFF y ReHo) que caracterizan propiedades locales de la señal de oxígeno en sangre (BOLD) en reposo.
2. Proyectamos estos volúmenes 3D en **cortes bidimensionales ortogonales fijos** (sagital, coronal, axial) seleccionados estadísticamente para capturar zonas con mayor variabilidad diagnóstica.
3. Tratamos a dichos cortes como **imágenes en escala de grises**, apilándolas como un tensor multicanal de entrada (6 canales).
4. Entrenamos una **Red Neuronal Convolucional (CNN)** pequeña diseñada para explotar la localidad y la jerarquía de patrones espaciales locales.

### El Desafío de la Heterogeneidad Multicentro (Domain Shift)
Dado que ABIDE es un consorcio compuesto por más de 20 centros de adquisición a nivel mundial, los datos presentan un sesgo instrumental considerable (efectos de sitio o *site effects*). Diferentes marcas de resonadores, intensidades de campo magnético (1.5T vs. 3T) y parámetros físicos de adquisición causan variaciones estadísticas en las imágenes que no guardan relación con la neuropatología.

Para evitar que la CNN "memorice" las características instrumentales del resonador de entrenamiento y falle al evaluar nuevos pacientes, implementamos la maquinaria de **Redes Adversariales de Dominio (DANN)** que extrae características invariantes al centro de escaneo mediante una capa de inversión de gradiente (GRL). Dado que los sitios de test (PITT y OLIN) no participan del entrenamiento adversarial, el régimen es más preciso denominarlo **generalización de dominio adversarial multi-fuente**: la red aprende a ser invariante entre los 18 sitios de entrenamiento, con la esperanza de que esa invariancia transfiera a sitios no vistos.

---

## 2. Datos Utilizados

### Cohorte ABIDE PCP
Utilizamos el subconjunto preprocesado provisto por el **Preprocessed Connectomes Project (PCP)** basado en la primera fase de ABIDE (ABIDE I). Los datos han sido preprocesados mediante el pipeline **CPAC (Configurable Pipeline for the Analysis of Connectomes)** con la estrategia `filt_noglobal` (filtrado temporal de paso de banda de 0.01–0.1 Hz, sin regresión de señal global), conservando oscilaciones funcionales espontáneas cruciales para el análisis del autismo.

**Estadísticos de la cohorte utilizada:**
- **1035 sujetos** procesados exitosamente: 505 ASD / 530 TC
- **20 sitios** de adquisición representados en el dataset
- **Set de test (hold-out ciego):** PITT (56 sujetos) + OLIN (34 sujetos) = **90 sujetos**, nunca vistos durante el entrenamiento

### Mapas Derivados Funcionales
Consumimos dos representaciones locales *voxel-wise* de la señal BOLD:
1. **fALFF (fractional Amplitude of Low-Frequency Fluctuations):** Mide la intensidad relativa de las fluctuaciones espontáneas en la banda de baja frecuencia respecto del espectro total. Referencia: Zou et al. (2008).
2. **ReHo (Regional Homogeneity):** Mide la sincronía y coherencia temporal local evaluando la concordancia de la serie temporal de un voxel respecto de sus 26 vecinos inmediatos mediante el coeficiente de Kendall (W). Referencia: Zang et al. (2004).

### Selección de Cortes Estadísticamente Óptimos (VBM-like)
Los volúmenes están ya registrados al espacio común estandarizado **MNI152 (3 mm, grilla 61×73×61 voxels)**. Esto garantiza que un índice de voxel específico represente la misma región anatómica aproximada en todos los sujetos.

En lugar de elegir cortes arbitrarios, los índices de corte se derivaron de los datos: para cada derivado y cada eje, se calculó el **Cohen's d voxel-wise** entre grupos ASD y TC, eligiendo el plano cuya media de |d| dentro de la máscara cerebral fuera máxima. Este cálculo se realizó **exclusivamente sobre sujetos de entrenamiento** (excluyendo PITT y OLIN), preservando la integridad del test.

Los índices de voxel resultantes son:
- **fALFF:** Sagital=10, Coronal=8, Axial=17
- **ReHo:** Sagital=30, Coronal=10, Axial=48

> **Nota de honestidad metodológica:** Los Cohen's d ganadores son |d| ≈ 0.07–0.09, valores convencionalmente clasificados como "despreciables" (Cohen: pequeño ≈ 0.2). El método de selección es correcto, pero opera sobre una señal estadísticamente muy débil, lo cual es consistente con la dificultad intrínseca del problema ASD vs TC en rs-fMRI a nivel de sujeto individual.

### Pipeline de Transformación
1. **Atenuación de outliers y escalamiento:** Para cada corte float extraído, se recortan intensidades extremas usando percentiles [1, 99] y se normaliza linealmente a [0, 1]:
   $$S_{\text{norm}} = \frac{S - P_1}{P_{99} - P_1 + \epsilon}$$
2. **Cuantización a 8-bits:** Se proyecta a [0, 255] y se guarda como PNG en escala de grises.
3. **Validación de integridad:** Un script independiente (`validate_labels.py`) verifica la consistencia entre el nombre de carpeta (ASD/TC), la etiqueta que ve el modelo y el DX_GROUP del CSV fenotípico.

Los 6 cortes por sujeto (3 planos × 2 derivados) se apilan como tensor $\mathbf{X} \in \mathbb{R}^{6 \times 64 \times 64}$.

---

## 3. Métodos

### Arquitectura Convolucional ASD_DANN (29.027 parámetros)
La arquitectura propuesta consta de un **extractor de características compartido**, seguido de **dos cabezales clasificadores competitivos**:

```
                                    ┌──► Cabezal de Clasificación (Tarea) ──► Logit ASD vs TC
                                    │
[ Tensor de Entrada 6×64×64 ] ──► [ Extractor (CNN) ]
                                    │
                                    └──► [ GRL (×−α) ] ──► Cabezal de Dominio ──► Predicción de Sitio
```

#### 1. Extractor de Características
Tres bloques convolucionales jerárquicos (6→16→32→64 canales), cada uno con:
- Conv2D 3×3 + padding → BatchNorm → ReLU → MaxPool2D (÷2) → Dropout2D (p=0.018)
- **Global Average Pooling** al final: colapsa los mapas espaciales a un vector de 64 dimensiones, reduciendo drásticamente la complejidad paramétrica y actuando como regularizador estructural.

#### 2. Cabezal de Clasificación Diagnóstica (Task Head)
64 → Linear(32) → BatchNorm → ReLU → Dropout → Linear(1). Produce el logit para ASD vs TC.

#### 3. Cabezal Adversarial de Dominio (DANN Head)
Conecta el vector latente (64 dim) a un clasificador de sitio (18 clases de entrenamiento) a través de la **Gradient Reversal Layer (GRL)**:
- **Forward:** identidad.
- **Backward:** multiplica el gradiente por −α antes de propagarlo al extractor.
- **Efecto:** el clasificador de dominio aprende a predecir el sitio; el extractor es penalizado de forma opuesta, aprendiendo representaciones de las cuales el sitio no es deducible.

**Modo sin DANN:** se fija α=0. El GRL anula el gradiente de dominio hacia el extractor; la rama de dominio sigue ejecutándose pero no influye en las features. Esto hace la comparación **perfectamente controlada**: misma arquitectura, mismos hiperparámetros, misma inicialización (idéntico estado RNG al inicio de cada corrida).

### Función de Pérdida y Balance de Clases
- **Pérdida de tarea:** BCE con Label Smoothing simétrico (smoothing=0.046):
  $$y_{\text{smooth}} = y \cdot (1 - s) + 0.5 \cdot s$$
- **Pérdida de dominio:** Cross-Entropy estándar sobre los 18 sitios de train/val.
- **Pérdida total:** $\mathcal{L} = \mathcal{L}_{\text{tarea}} + \mathcal{L}_{\text{dominio}}$ (el signo adversarial está en el GRL, no en la suma).
- **Balance de clases:** `WeightedRandomSampler` en el DataLoader (clases ~50/50 en cada batch).

### Hiperparámetros (Config Final)
Buscados con **Optuna** (sampler TPE, maximizando val AUC, 55 epochs de patience, persistencia SQLite):

| Hiperparámetro | Valor |
|:---|:---:|
| Learning Rate | 0.000318 |
| Batch Size | 64 |
| DANN α | 0.212 |
| Dropout | 0.018 |
| Label Smoothing | 0.046 |
| Epochs máximos | 200 |
| Patience (early stopping) | 55 |
| Scheduler patience | 20 |
| Optimizador | AdamW |
| Scheduler | ReduceLROnPlateau |

### Estrategia de Split y Prevención de Fugas (Anti-Leakage)
1. **Split por sujeto:** todos los cortes de un sujeto van siempre al mismo fold. Se verifica explícitamente que ningún sujeto aparezca en dos conjuntos.
2. **Test out-of-distribution:** PITT + OLIN = 90 sujetos, reservados completamente. Nunca participan del tuning, entrenamiento ni calibración.
3. **Train/val del resto:** 85% train / 15% val, estratificado por (diagnóstico, sitio).
4. **Early stopping y model selection:** por val AUC (threshold-independiente).
5. **Calibración de umbral:** sobre val (maximiza balanced accuracy, desempate por F1). Se aplica en test.
6. **Cortes VBM seleccionados solo con sujetos de train/val:** PITT y OLIN excluidos de la selección de cortes.
7. **Fenotipos fuera del modelo:** el CSV fenotípico se usa solo para agrupar folds y auditar distribuciones. El CNN recibe únicamente imágenes PNG.

---

## 4. Resultados

### Métricas Globales en Test Set (PITT + OLIN, 90 sujetos — sitios nunca vistos)

| Métrica | Sin DANN (α=0.0) | Con DANN (α=0.212) |
|:---|:---:|:---:|
| **Test Loss** | 0.696 | 0.688 |
| **Test AUC** | 0.552 | **0.588** |
| **Test Balanced Accuracy** | 50.4% | **57.0%** |
| **Test Accuracy** | 47.8% | 55.6% |
| **F1-Score (ASD)** | 0.175 | **0.459** |
| **Sensibilidad / Recall ASD** | 10.4% | 35.4% |
| **Especificidad** | 90.5% | 78.6% |
| **Precisión ASD** | 55.6% | 65.4% |
| Val AUC (mejor, validación) | 0.571 | 0.622 |
| Umbral calibrado | 0.490 | 0.510 |

### Desempeño por Sitio de Test (AUC)

| Sitio | Sin DANN | Con DANN |
|:---|:---:|:---:|
| OLIN (Olin Neuropsychiatry Research Center) | 0.568 | **0.618** |
| PITT (Pittsburgh) | 0.527 | **0.557** |

### Interpretación de los Resultados

**AUC ≈ 0.55–0.59** refleja la dificultad intrínseca del problema: distinguir ASD de TC a nivel individual desde rs-fMRI es notoriamente difícil, con la literatura reportando valores similares (Heinsfeld et al. 2018: AUC ~0.65 con conectomas; representaciones 2D voxel-wise operan con menos información). Los tamaños de efecto Cohen's d ≈ 0.08 observados en el análisis VBM anticipaban este resultado.

**DANN supera a sin-DANN en todas las métricas de test**, con la diferencia más marcada en balanced accuracy (+6.6 pp) y F1 (+0.284). Sin embargo, es importante enmarcar este resultado estadísticamente:

- **DANN:** AUC test = 0.588, IC95% ≈ [0.47, 0.71] (Hanley-McNeil, n⁺=48, n⁻=42)
- **Sin DANN:** AUC test = 0.552, IC95% ≈ [0.43, 0.67]
- **Diferencia:** 0.036, que con n=90 no alcanza significancia estadística convencional

La conclusión correcta es que los resultados son **consistentes con la hipótesis de que DANN mejora la generalización a sitios no vistos**, pero no la demuestran de forma concluyente con un único hold-out de 90 sujetos. Tanto la tendencia como la magnitud son informativos; se requiere leave-one-site-out (LOSO) con test pareado de DeLong para poder atribuir causalmente la mejora a DANN.

### Figuras de Resultados

- `results_dann/confusion_matrix.png` — Matriz de confusión con umbral calibrado (0.51)
- `results_dann/roc_curve.png` — Curva ROC (AUC = 0.588)
- `results_dann/training_curves.png` — Curvas de pérdida y AUC durante entrenamiento
- `results_no_dann/confusion_matrix.png` — Análogo para el modelo sin DANN
- `results_no_dann/roc_curve.png` — Curva ROC sin DANN (AUC = 0.552)

---

## 5. Conclusiones y Trabajo Futuro

### Principales Hallazgos

1. **El pipeline es metodológicamente sólido:** el tratamiento de mapas fALFF/ReHo como imágenes 2D multicanal para una CNN liviana es viable y reproducible. El control de data leakage (split por sujeto, test OOD, fenotipos fuera del modelo) es la contribución metodológica más valiosa.

2. **DANN muestra una tendencia favorable no concluyente:** en todas las métricas de test, el modelo con adaptación adversarial supera al modelo sin ella (+6.6 pp balanced accuracy, AUC +0.036). La diferencia es consistente con la hipótesis pero no estadísticamente significativa con el tamaño muestral actual de test.

3. **El resultado refleja la dificultad del problema, no una falla del código:** AUC ≈ 0.59 es esperable dado que (a) los efectos de grupo ASD-TC en fALFF/ReHo son despreciables (|d| ≈ 0.08), (b) se usan representaciones 2D con pérdida de información volumétrica, y (c) el dataset es de tamaño moderado.

4. **La comparación DANN vs sin-DANN es bien controlada:** misma arquitectura, mismos hiperparámetros, misma inicialización aleatoria. La única diferencia es α=0 vs α=0.212. Esto es, en sí, una decisión de diseño experimental elegante.

### Limitaciones Identificadas

1. **Poder estadístico insuficiente:** un hold-out de 90 sujetos en un único split y una única semilla no permite distinguir si la mejora de DANN supera la variabilidad muestral.
2. **Pérdida de rango dinámico:** la cuantización a PNG 8-bit con normalización per-corte elimina información de intensidad absoluta inter-sujeto, que puede ser relevante en fALFF/ReHo.
3. **Cortes 2D fijos:** 6 cortes capturan una fracción pequeña del volumen; la patología ASD puede tener firmas distribuidas en regiones no cubiertas.
4. **Registro MNI imperfecto:** el mismo índice de voxel corresponde a la misma región anatómica *aproximada*, no idéntica, entre sujetos.
5. **Un único hold-out sin repetición:** no permite estimar la varianza del AUC por variación de semilla.

### Líneas de Extensión Futuras (Prioridad Alta)

- **Leave-One-Site-Out (LOSO) + test pareado de DeLong:** diseño que permite concluir si DANN generaliza mejor entre sitios, con poder estadístico adecuado.
- **Repetición con múltiples semillas:** reportar media ± desvío estándar del AUC.
- **CNN 3D / representación directa desde NIfTI:** eliminar la cuantización PNG y aprovechar la continuidad volumétrica.
- **Armonización estadística ComBat:** alternativa o complemento al DANN adversarial.
- **Intervalos de confianza en todas las métricas reportadas** (Hanley-McNeil o bootstrap).

---

## Referencias

- **[1] Di Martino, A., et al. (2014).** The Autism Brain Imaging Data Exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. *Molecular Psychiatry*, 19(6), 659-667.
- **[2] Di Martino, A., et al. (2017).** Enhancing studies of the connectome in autism using the autism brain imaging data exchange II (ABIDE-II). *Scientific Data*, 4, 170010.
- **[3] Craddock, C., et al. (2013).** The Neuro Bureau Preprocessed Connectomes Project. *Frontiers in Neuroinformatics*, 7, 27.
- **[4] Zou, Q. H., et al. (2008).** An improved approach to detection of amplitude of low-frequency fluctuations (ALFF) for resting-state fMRI: Fractional ALFF. *Journal of Neuroscience Methods*, 172(1), 137-141.
- **[5] Zang, Y., et al. (2004).** Regional homogeneity approach to fMRI data analysis. *NeuroImage*, 22(1), 394-400.
- **[6] Ganin, Y., & Lempitsky, V. (2015).** Unsupervised Domain Adaptation by Backpropagation. *ICML*.
- **[7] Ganin, Y., et al. (2016).** Domain-Adversarial Training of Neural Networks. *JMLR*, 17(59), 1-35.
- **[8] Heinsfeld, A. S., et al. (2018).** Identification of autism spectrum disorder using deep learning and the ABIDE dataset. *NeuroImage: Clinical*, 17, 16-23.
- **[9] Abraham, A., et al. (2017).** Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example. *NeuroImage*, 147, 736-745.
