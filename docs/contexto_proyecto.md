# Contexto del proyecto TP Computer Vision — ABIDE

Documento de contexto técnico del proyecto: qué es, cómo encajan los datos y las decisiones de diseño.

---

## 1. Qué es este proyecto (en pocas palabras)

Es un **trabajo práctico de visión por computadora** sobre datos públicos de neuroimagen (**ABIDE**): personas con diagnóstico de **autismo (ASD)** frente a **controles típicos (TC)**. En lugar de usar matrices de conectividad (más cercanas a grafos/tablas), el enfoque es representar **mapas funcionales del cerebro en reposo** como **imágenes en escala de grises** y entrenar una **CNN pequeña** para una tarea de **clasificación binaria**.

No es un sistema de diagnóstico clínico: es un **pipeline reproducible** (datos → imágenes → modelo) alineado con literatura y buenas prácticas mínimas de evaluación.

---

## 2. Objetivos

| Objetivo | Detalle breve |
|----------|----------------|
| **Pedagógico** | Aplicar herramientas de visión (CNN, splits, métricas) sobre entradas tipo imagen. |
| **Técnico** | Consumir derivados **CPAC** ya publicados (fALFF, ReHo), proyectarlos a 2D y clasificar ASD vs TC. |
| **Metodológico** | Documentar limitaciones (MNI, PNG, multi-sitio, tamaño muestral). |

---

## 3. Cómo las “mediciones cerebrales” pasan a imágenes en escala de grises

### 3.1 Formato de las mediciones

Lo que se descarga del almacenamiento público son volúmenes **NIfTI** (`.nii.gz`): un **arreglo tridimensional de números reales** (float) en una rejilla de voxels. En este proyecto esos volúmenes ya están en espacio **MNI** y corresponden a **mapas derivados** del preproceso (no son la serie BOLD 4D cruda en el repo).

### 3.2 Qué se hace en el código del TP

1. Se carga el volumen 3D con **NiBabel** (`get_fdata()`).
2. Se extraen **tres cortes 2D fijos** por volumen (sagital, coronal, axial) usando **índices de voxel** fijos en esa rejilla MNI (ver `scripts/downloader.py`).
3. Cada matriz 2D se **normaliza para visualización**: percentiles 1 y 99, recorte, escalado a 0–255, guardado como **PNG en escala de grises**.

Es decir: no es que la resonancia “sea” una foto; es un **mapa numérico** que se **trata como imagen** para alimentar redes convolucionales (enfoque habitual en visión médica con mapas paramétricos).

---

## 4. CPAC: dónde ocurre, entrada y salida; fALFF y ReHo

**CPAC** (Configurable Pipeline for the Analysis of Connectomes) es un **pipeline de preproceso** que se ejecutó **fuera de este repositorio**, en el marco de **ABIDE Preprocessed** (PCP). Este proyecto **no vuelve a correr CPAC**; **consume los productos** publicados (por ejemplo en S3) con una estrategia fija (`filt_noglobal` en el código actual).

**Entrada conceptual del pipeline upstream:** datos funcionales (típicamente **BOLD** en reposo) ya sometidos a pasos de corrección, normalización espacial, etc., según la documentación PCP/CPAC.

**Salida que usa este TP:** volúmenes 3D por sujeto y derivado, en particular:

| Derivado | Idea intuitiva | Tipo de dato |
|----------|----------------|--------------|
| **fALFF** | Mapa relacionado con la **amplitud de fluctuaciones de baja frecuencia** en la señal BOLD, resumida en el tiempo (fraccional respecto del espectro). | Volumen 3D float por voxel |
| **ReHo** | Mapa de **homogeneidad regional**: qué tan similares son las series temporales de un voxel respecto a sus vecinos. | Volumen 3D float por voxel |

Referencias metodológicas y de cohorte: [referencias.md](referencias.md).

---

## 5. “Física” del volumen 3D, cortes y comparabilidad entre sujetos

Los volúmenes están en **espacio MNI** (plantilla común): cada cerebro fue registrado/normalizado al mismo sistema de coordenadas. Por eso **el mismo índice de voxel** en el mismo pipeline corresponde a **la misma región anatómica aproximada** entre sujetos, no al mismo milímetro en el cráneo nativo.

**Cortes sagital / coronal / axial en índices fijos:** son simples y comparables en MNI; **no** son la única opción posible. Alternativas según el problema: más cortes, proyecciones (promedio, MIP), coordenadas explícitas en mm, superficie cortical, tensores multi-canal, etc.

**Limitación honesta:** el registro a MNI no es perfecto; la anatomía individual varía. Conviene mencionarlo en el informe.

---

## 6. Buena práctica: train/validation por sujeto (no por PNG)

Cada persona genera **varios** archivos PNG (vistas × derivados). Si el split train/val se hiciera **mezclando archivos** sin agrupar por sujeto, podría ocurrir que **el mismo individuo** tenga un corte en entrenamiento y otro en validación. El modelo podría aprender **rasgos idiosincráticos de esa persona** y la métrica de validación se **inflaría** (fuga de información / *leakage*).

El pipeline parte los conjuntos **por sujeto**: `scripts/dataset.py` agrupa los 6 PNGs de cada sujeto y `scripts/train.py` reserva además **sitios completos** (PITT/OLIN) como test. Así, todo lo de un sujeto queda solo en train, val **o** test.

---

## 7. Formato de descarga y uso de PNG

- **Descarga:** archivos **NIfTI** (`.nii.gz`), formato estándar en neuroimagen.
- **PNG:** cuantización a **8 bits** tras normalización por percentiles; **pérdida de rango dinámico** respecto del float original. Es **cómodo** para prototipos y CNNs 2D; la **fuente de verdad** para re-procesar cortes o normalizaciones distintas sigue siendo el **NIfTI** cacheado bajo `abide_data/`.

**Evidencia / defensa bibliográfica:** citar definiciones de fALFF/ReHo, preproceso PCP/CPAC y trabajos de deep learning sobre ABIDE; declarar el PNG como **representación 2D derivada** con limitaciones (ver [referencias.md](referencias.md)).

---

## 8. Descargar vs convertir: qué hace el pipeline actual

Por sujeto, el flujo es: **descargar NIfTI si no existe → leer volumen → generar PNG en el mismo paso**. No hay un paso manual intermedio obligatorio. Guardar NIfTI permite **no re-descargar** y **cambiar** cortes o normalización después sin volver a S3.

---

## 9. ABIDE I vs ABIDE II y datos fenotípicos (estado y deseo)

**Estado actual del código:** el CSV fenotípico y las URLs de derivados están alineados con el ecosistema típico **ABIDE I / PCP** (incluyendo el bucket S3 usado en `scripts/downloader.py`). **ABIDE II no está integrado** en el descargador ni en una base unificada.

**Deseable (comentado para implementación futura):**

- Incorporar **ABIDE II** además de ABIDE I (o justificar solo una cohorte).
- Construir una **base tabular** por sujeto con **demografía, lenguaje, funciones cognitivas, sitio / resonador, QC**, y demás columnas disponibles en los CSV oficiales, con documentación de **missingness** y de **homogeneización de nombres** entre cohortes.

---

## 10. Enlaces rápidos en el repo

| Recurso | Uso |
|---------|-----|
| [README.md](../README.md) | Comandos (`uv`, `main.py`, entrenamiento). |
| [referencias.md](referencias.md) | Papers ancla y orden de lectura. |
| [cortes_mni_y_mapas_derivados.md](cortes_mni_y_mapas_derivados.md) | Índices de corte MNI, mapas fALFF/ReHo y límites del PNG. |
| `scripts/downloader.py` | Descarga NIfTI, genera PNGs. |
| `scripts/train.py` | Entrenamiento DANN; split por sujeto/sitio; CLI por modo (`--mode dann/no-dann`). |
| `scripts/compare_dann.py` | Entrena y compara DANN vs sin-DANN desde `config.toml`. |

---

*Documento de contexto técnico del proyecto.*
