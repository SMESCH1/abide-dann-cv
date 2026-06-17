# Papers y recursos ancla — ABIDE, visión por computadora, rs-fMRI

Lista curada para citar en un TP de **visión por computadora** sobre **ASD vs. control** con datos **ABIDE** y representaciones tipo imagen (cortes / mapas derivados). Incluye cohorte, preproceso, definición de métricas, deep learning y evaluación multi-sitio.

## Prioridad de lectura (para entender el TP)

**Qué es rs-fMRI en ABIDE y para qué sirve en estudios como este**

1. **Di Martino *et al.* (2014)** — contexto de la cohorte: qué datos comparten, reposo, multi-sitio; lectura relativamente corta y obligatoria para enmarcar el TP.

2. **Craddock *et al.* (2013) + página ABIDE Preprocessed** — de dónde salen los volúmenes ya preprocesados (CPAC, estrategias) y por qué existen derivados homogéneos como fALFF/ReHo.

**Qué miden fALFF y ReHo (mapas que luego cortás a 2D)**

3. **Zou *et al.* (2008)** (fALFF) y **Zang *et al.* (2004)** (ReHo) — definiciones originales; con leer introducción + métodos alcanza para el informe.

4. **Zuo *et al.* (2013)** — opcional pero útil: ALFF/fALFF en datos multi-sitio (cautela con comparabilidad).

**Deep learning + límites (cuando conectes con la CNN)**

5. **Heinsfeld *et al.* (2018, Frontiers Neuroscience, CNN)** o **Heinsfeld *et al.* (2018, NPJ Sci Learn)** — uno basta como ancla de “DL + ABIDE”; el segundo si querés comparar enfoques.

6. **Plis *et al.* (2021)** — revisión accesible: qué se ha hecho y qué problemas son recurrentes (tamaño muestral, generalización).

7. **Abraham *et al.* (2017)** o **Dong *et al.* (2025)** — cuando documentes el split y el efecto de sitio: buenas prácticas y comparación de clasificadores.

---

## 1. Cohorte y compartición de datos

| Recurso | Por qué citarlo |
|--------|------------------|
| **Di Martino, A., *et al.* (2014).** *The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism.* **Molecular Psychiatry**, 19(6), 659–667. [DOI 10.1038/mp.2013.78](https://doi.org/10.1038/mp.2013.78) · [PubMed 23774715](https://pubmed.ncbi.nlm.nih.gov/23774715/) | Paper fundacional de **ABIDE I**: diseño multi-sitio, N, tipo de datos (rs-fMRI + fenotipos). |
| **Di Martino, A., *et al.* (2017).** *Enhancing studies of the connectome in autism using the autism brain imaging data exchange II.* **Scientific Data** 4, 170010. [DOI 10.1038/sdata.2017.10](https://doi.org/10.1038/sdata.2017.10) | Descriptor de **ABIDE II**; útil si comparan o citan ampliaciones de la cohorte. |

---

## 2. Preproceso compartido (PCP / CPAC / Nilearn)

| Recurso | Por qué citarlo |
|--------|------------------|
| **Craddock, C., *et al.* (2013).** *The Neuro Bureau Preprocessing Initiative: open sharing of preprocessed neuroimaging data and derivatives.* In **Neuroinformatics 2013**, Stockholm (abstract Frontiers). [Resumen](https://www.frontiersin.org/10.3389/conf.fninf.2013.09.00041/event_abstract) · [ABIDE Preprocessed](http://preprocessed-connectomes-project.org/abide/) | Cita oficial al usar **ABIDE Preprocessed** (varios pipelines: CPAC, CCS, DPARSF, NIAK; derivados homogeneizados). |
| **Nilearn — ABIDE PCP.** [Documentación `fetch_abide_pcp`](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_abide_pcp.html) | Encaje con el código Python que arranca la descarga de fenotipos / rutas PCP. |

---

## 3. Definición de métricas de mapa (fALFF, ReHo)

Justifican que los volúmenes **fALFF** y **ReHo** sean mapas funcionales estándar antes de proyectarlos a 2D.

| Recurso | Por qué citarlo |
|--------|------------------|
| **Zang, Y., *et al.* (2004).** *Regional homogeneity approach to fMRI data analysis.* **NeuroImage** 22(1), 394–400. | Introduce **ReHo** (homogeneidad regional). |
| **Zou, Q.-H., *et al.* (2008).** *An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF.* **Journal of Neuroscience Methods** 172(1), 137–141. [PubMed 18501969](https://pubmed.ncbi.nlm.nih.gov/18501969/) | Define **fALFF** como alternativa a ALFF. |
| **Zuo, X.-N., *et al.* (2013).** *A multi-site resting-state fMRI study on the amplitude of low frequency fluctuations in schizophrenia.* **Frontiers in Neuroscience** 7, 137. [PMC3737471](https://pmc.ncbi.nlm.nih.gov/articles/PMC3737471/) | Ejemplo multi-sitio con **ALFF/fALFF** en controles y pacientes; útil para argumentar uso de mapas de amplitud en cohortes heterogéneas. |

---

## 4. Deep learning + ABIDE (entradas neuroimagen)

| Recurso | Por qué citarlo |
|--------|------------------|
| **Heinsfeld, A. S., *et al.* (2018).** *Identification of autism spectrum disorder using deep learning and the ABIDE dataset.* **Nature Partner Journals Science of Learning** 3, 19. [PMC5635344](https://pmc.ncbi.nlm.nih.gov/articles/PMC5635344/) | DL sobre ABIDE; discusión relevante para **generalización** y enfoque data-driven. |
| **Heinsfeld, A. S., *et al.* (2018).** *Automated detection of autism spectrum disorder using a convolutional neural network.* **Frontiers in Neuroscience** 12, 517. [PMC6971220](https://pmc.ncbi.nlm.nih.gov/articles/PMC6971220/) | **CNN** explícita para detección ASD con datos ABIDE. |
| **Jiang, W., *et al.* (2022).** *CNNG: A convolutional neural networks with gated recurrent units for autism spectrum disorder classification.* **Frontiers in Aging Neuroscience** 14, 948704. [DOI 10.3389/fnagi.2022.948704](https://doi.org/10.3389/fnagi.2022.948704) · [PMC9294312](https://pmc.ncbi.nlm.nih.gov/articles/PMC9294312/) · [PubMed 35865746](https://pubmed.ncbi.nlm.nih.gov/35865746/) | **3D CNN + GRU** sobre fMRI ABIDE (espacio + temporalidad en la serie). |
| **Epalle, T. M., *et al.* (2021).** *Multi-atlas classification of autism spectrum disorder with hinge loss trained deep architectures: ABIDE I results.* **Neurocomputing** 458, 757–769. [DOI 10.1016/j.neucom.2021.08.005](https://doi.org/10.1016/j.neucom.2021.08.005) | DL multi-atlas sobre ABIDE; útil si discuten **robustez** a la parcelación. |
| **Arya, A., *et al.* (2020).** *Fusing structural and functional MRIs using graph convolutional networks for autism classification.* **Proceedings of Machine Learning Research** (ML4H), 124, 169–181. [PMLR](http://proceedings.mlr.press/v121/arya20a/arya20a.pdf) | Multimodal estructura–función con GCN (comparar con enfoque puramente “imagen”). |

---

## 5. Evaluación multi-sitio, revisiones y límites

| Recurso | Por qué citarlo |
|--------|------------------|
| **Abraham, A., *et al.* (2017).** *Deriving reproducible biomarkers from multi-site resting-state data: An autism-based example.* **NeuroImage** 147, 736–745. [DOI 10.1016/j.neuroimage.2016.10.045](https://doi.org/10.1016/j.neuroimage.2016.10.045) · [PubMed 27865923](https://pubmed.ncbi.nlm.nih.gov/27865923/) | **ComBat** y biomarcadores rs-fMRI **reproducibles** en datos multi-sitio con ejemplo ASD. |
| **Dong, Y., *et al.* (2025).** *A framework for comparison and interpretation of machine learning classifiers to predict autism on the ABIDE dataset.* **Human Brain Mapping** 46(5), e70190. [DOI 10.1002/hbm.70190](https://doi.org/10.1002/hbm.70190) · [PubMed 40095417](https://pubmed.ncbi.nlm.nih.gov/40095417/) · [PMC11912182](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912182/) | Marco comparativo de clasificadores + **interpretación** en ABIDE; código relacionado en [Machine-learning-with-ABIDE](https://github.com/YilanDong19/Machine-learning-with-ABIDE). |
| **Bzdok, D., & Meyer-Lindenberg, A. (2018).** *Machine learning for precision psychiatry: opportunities and challenges.* **Biological Psychiatry: Cognitive Neuroscience and Neuroimaging** 3(3), 223–230. | Crítica de **overfitting** y promesas de “psychiatry AI” (contexto para el informe). |
| **Plis, S. M., *et al.* (2021).** *Neuroimaging-based deep learning in autism spectrum disorder and attention-deficit/hyperactivity disorder.* **Biological Psychiatry: Cognitive Neuroscience and Neuroimaging** 6(9), 864–877. [PMC7350542](https://pmc.ncbi.nlm.nih.gov/articles/PMC7350542/) | **Revisión** DL + neuroimagen en TDAH/ASD; tabla de estudios y limitaciones. |
| **Vidya, M. K., *et al.* (2025).** *Identification of critical brain regions for autism diagnosis from fMRI data using explainable AI: an observational analysis of the ABIDE dataset.* **eClinicalMedicine** (The Lancet Discovery Science). [Artículo](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(25)00384-0/fulltext) | DL + **XAI** + controles de calidad / movimiento en ABIDE (muy reciente). |

---

## 6. Buenas prácticas de réplica (checklist breve)

- Citar **cohorte (Di Martino)** + **ABIDE Preprocessed (Craddock *et al.*)** + pipeline concreto (**CPAC**, estrategia `filt_noglobal` o la que usen).
- Declarar **split** (p. ej. k-fold, o hold-out por **sitio** cuando sea posible).
- Mencionar **efecto de sitio** y, si aplica, harmonización (p. ej. ComBat en Abraham *et al.*).
- Separar claramente **mapa estadístico 2D** (input del modelo) de **claims clínicos** (Vidya *et al.* y revisiones enfatizan cautela).

*Última actualización: alineado al plan del TP y al script `scripts/downloader.py` (derivados fALFF/ReHo, CPAC).*
