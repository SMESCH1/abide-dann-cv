# Cortes en espacio MNI y mapas derivados (fALFF / ReHo)

Documento breve para **defensa metodológica** y para guiar experimentos (p. ej. cambiar índices de corte o ROI).

## Qué es «funcional» en este proyecto

Los volúmenes que descarga `scripts/downloader.py` **no** son anatomía T1 clásica: son **mapas 3D en espacio MNI** donde cada voxel resume información derivada de la señal **BOLD en reposo** tras el preproceso **ABIDE Preprocessed / CPAC** (estrategia `filt_noglobal`). Es decir: **valor por voxel = resumen estadístico funcional**, ya compactado en una imagen 3D por sujeto.

- **fALFF**: amplitud fraccional de fluctuaciones de baja frecuencia (definición metodológica: Zou et al., 2008).
- **ReHo**: homogeneidad regional / similitud local de series temporales con vecinos (Zang et al., 2004).

No confundir con **redes funcionales** en el sentido de conectividad entre ROI (series correlacionadas entre regiones): aquí la entrada a la CNN es un **campo escalar por voxel** (un «mapa»), no una matriz de conectividad.

## Qué rejilla e índices usa el código

En `scripts/downloader.py` los cortes 2D se obtienen por **índices enteros** sobre el volumen **61 × 73 × 61** (rejilla **MNI 3 mm** del producto PCP usado). Desde el merge `70550ca` los índices son **VBM-óptimos por derivado** (no fijos heredados del MVP).

### Índices VBM-óptimos (vigentes)

`statistical_analysis/vbm_analysis.py` construye una **máscara cerebral**, calcula el **tamaño de efecto (Cohen's d) voxel-a-voxel** ASD vs TC **excluyendo los sitios PITT y OLIN** (reservados como test set para evitar leakage al elegir cortes), y por cada eje elige el plano con mayor **media de `|d|`** dentro de la máscara.

| Derivado | Sagital | Coronal | Axial | media \|d\| (sag/cor/ax) |
|----------|---------|---------|-------|--------------------------|
| fALFF    | `d[57,:,:]` | `d[:,8,:]` | `d[:,:,17]` | 0.069 / 0.089 / 0.072 |
| ReHo     | `d[30,:,:]` | `d[:,10,:]` | `d[:,:,60]` | 0.074 / 0.080 / 0.067 |

Resultados completos en `statistical_analysis/optimal_slices.json`.

### Índices legacy (deprecados, conservados como referencia)

Versiones del downloader anteriores al merge `70550ca` (branch `feat/phenotypic-manifest`, MVP inicial) usaban cortes fijos arbitrarios, **iguales para fALFF y ReHo**:

| Vista | Índice legacy | Lambda |
|-------|---------------|--------|
| Sagital | 30 | `d[30, :, :]` |
| Coronal | 36 | `d[:, 36, :]` |
| Axial | 30 | `d[:, :, 30]` |

Elección heurística: cerca del centro del volumen, sin justificación estadística.

Las PNGs generadas con estos cortes están preservadas en `dataset/legacy_slices_fixed_30-36-30/` (gitignored como el resto de `dataset/*`) — útiles para una comparativa empírica VBM vs arbitrario y para ablations en el informe. Más detalle: `dataset/legacy_slices_fixed_30-36-30/SLICES_INFO.md`.

Los índices son **coordenadas en índices de matriz**, no milímetros MNI directos.

## Por qué «el mismo índice» tiene sentido entre sujetos

Tras normalización a **espacio MNI**, **el mismo índice (i, j, k)** en la misma rejilla del pipeline apunta a **la misma ubicación anatómica aproximada** entre sujetos (comparabilidad espacial). Esto es estándar en neuroimagen colectiva y es lo que permite tratar cortes fijos como entrada comparable para una CNN 2D (con las salvedades de registro imperfecto y variabilidad individual).

Referencias de contexto: cohorte y uso de datos compartidos (**Di Martino et al., 2014**); productos preprocesados y derivados homogéneos (**Craddock et al., 2013** / ABIDE Preprocessed). Detalle de definiciones de mapas y lecturas sugeridas: [referencias.md](referencias.md).

## Por qué están los cortes *ahí* (literatura vs ingeniería)

**Importante:** la literatura cita **regiones**, **coordenadas MNI en mm**, meta-análisis o ROI cuando estudia diferencias grupo ASD vs control. **No** suele prescribir «usar el índice voxel 30 axial para clasificación».

En este TP, la elección **vigente** de índices VBM-óptimos por derivado es una **decisión data-driven con cuidado anti-leakage**: el análisis que selecciona el corte (Cohen's d voxel-a-voxel) excluye los sitios test (PITT, OLIN), de modo que la elección no «mira» los datos sobre los que después se evalúa. El alternativo histórico — cortes fijos centrales — fue una **decisión de ingeniería** del MVP (reproducible, barato), sin garantía de capturar señal discriminativa.

Para argumentar que un corte coincide con regiones de interés (p. ej. redes relacionadas con lenguaje o sociocognición), el siguiente paso metodológico sería **traducir coordenadas o máscaras de atlas a (i, j, k)** en **esta** rejilla, o **recortar / promediar** un entorno alrededor de un ROI.

## Limitaciones del PNG

El paso `to_png` recorta por percentiles 1–99 y cuantiza a 8 bits: es una **representación visual** conveniente; la fuente numérica completa sigue en el `.nii.gz` bajo `abide_data/`.

## Lecturas directas en el repo

- Tabla de índices: `scripts/downloader.py` (`SLICES`).
- Contexto ampliado del TP: [contexto_proyecto.md](contexto_proyecto.md).
- Papers ancla: [referencias.md](referencias.md).
