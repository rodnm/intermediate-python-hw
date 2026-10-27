# Tarea 2 – Voces y documentos peruanos: del archivo al texto buscable

> **Pregunta guía:** ¿Pueden convertir documentos escaneados y entrevistas en audio en un archivo de texto estructurado y buscable?

**Respuesta corta: sí, con matices según el tipo de fuente.** Se procesaron 25 documentos y 40.6 minutos de audio en español peruano. Todo quedó como texto buscable, seudonimizado y con campos estructurados.

| Modalidad | Línea base | Modelo final | Resultado |
|---|---|---|---|
| Documentos (CER) | PaddleOCR: 0.31 | Gemini como modelo de visión y lenguaje | **0.14** |
| Audio (WER) | faster-whisper `small`: 0.22 | Parakeet TDT 0.6B v3 | **0.14** |
| Extracción con LLM | — | Gemini, con esquema JSON | **11/20** extracciones completamente correctas en la auditoría |

El punto débil son los manuscritos para el OCR local y la calle con varias voces para el ASR.

| | |
| :--- | :--- |
| **Autor** | Rodrigo Andrés Norabuena Mascaraqui ([@rodnm](https://github.com/rodnm)) |
| **Curso** | Python Intermedio · Diplomatura de especialización Ciencia de Datos para las Ciencias Sociales y la Gestión Pública 2026 |
| **App desplegada** | <https://voces-documentos-rodnm.streamlit.app> |
| **Enunciado** | [d2cml-ai/Data-Science-Python#190](https://github.com/d2cml-ai/Data-Science-Python/issues/190) |
| **Fuentes** | Wikimedia Commons y OpenSLR 73 (todo con licencia abierta) |

```
Resultados Test: WER: 0.14 | CER: 0.14 | Acuerdo LLM-Humano: 11/20
```

---

## 1. Cómo ejecutar en local

El dashboard solo lee los CSV de `datos/` y `modelos/metricas.json`: no corre OCR, ASR ni LLM en vivo.

```bash
uv venv --python 3.12
uv pip install -r tarea_2/requirements.txt
streamlit run tarea_2/app.py      # desde la raíz del repositorio, para que cargue el tema de .streamlit/
```

**Notebooks:** el entorno completo y congelado está en `requirements-notebooks.txt` (PaddleOCR, faster-whisper, pyannote.audio, transformers, torch…). La extracción con LLM necesita `GEMINI_API_KEY` en un archivo `tarea_2/.env`, que no se versiona.

```bash
uv venv tarea_2/.venv --python 3.12
uv pip install --python tarea_2/.venv/Scripts/python.exe -r tarea_2/requirements-notebooks.txt
cd tarea_2 && .venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace 01_extraccion.ipynb 02_evaluacion.ipynb
```

- Los notebooks usan la caché de `datos/raw/` (`REPROCESAR = False`). Esa carpeta no se versiona: con `REPROCESAR = True`, el notebook 01 vuelve a descargar todo desde `fuentes.csv`.
- **Orden:** primero el 01, después el 02. El 02 completa `texto_referencia` en los CSV que escribe el 01.

## 2. Corpus y fuentes

Fecha de extracción: **26/10/2026**. Detalle por archivo (licencia, autor y cómo se recupera): [`datos/fuentes.csv`](datos/fuentes.csv) y [`datos/atribucion.csv`](datos/atribucion.csv).

| Grupo | Cantidad | Fuente | Licencias |
|---|---|---|---|
| Impresos nítidos | 8 | Normas, comunicados y boletines en Commons | Dominio público, CC BY 4.0 |
| Fotos con sombra o inclinación | 8 | Fotos de impresos en Commons (EXIF de cámara o celular) | CC0, CC BY 2.0, CC BY-SA 4.0 |
| Manuscritos | 9 | Cartas y notas de los siglos XIX y XX en Commons (BNP y colección Fitte) | Dominio público, CC0 |
| Entrevistas | 8 grabaciones → 33 clips de ≤ 60 s (27.3 min) | Commons, en español peruano | CC BY-SA 4.0 |
| Lectura | 120 oraciones (13.4 min) de 30 hablantes | [OpenSLR 73](https://www.openslr.org/73/) (Google) | CC BY-SA 4.0 |

**Procedimiento:**
- **Descarga respetuosa:**
  - User-Agent descriptivo, pausa de 1–2 s y `robots.txt` revisado por host;
  - OpenSLR se lee por HTTP Range, sin bajar los zips de ~1 GB.
- **Revisión visual previa:** antes de sortear la muestra se descartaron 5 documentos y 2 reservas, que se reemplazaron:
  - fotos casi sin texto;
  - cartas mecanografiadas que solo tenían la firma a mano.

  El detalle está en [`datos/descartes.csv`](datos/descartes.csv).
- **Fuera del corpus, por licencia o por falta de referencia:**
  - el debate presidencial de 2006, una grabación de TV con licencia dudosa;
  - los audios en quechua de Commons, que no tienen transcripción humana.

<details>
<summary>Tabla por archivo</summary>

| id | Tipo | Documento | Licencia | Enlace |
|---|---|---|---|---|
| doc_001 | impreso | Decreto Supremo 001-81-AA que declara el Santuario Histórico de Machu Picchu (El Peruano, 1981), p. 1 | CC BY 4.0 | [Commons](https://commons.wikimedia.org/?curid=179771570) |
| doc_002 | impreso | Decreto Supremo 046-2020-PCM sobre el toque de queda por COVID-19, p. 1 | Dominio público | [Commons](https://commons.wikimedia.org/?curid=88315184) |
| doc_003 | impreso | Decreto de Urgencia 002-2019 sobre las elecciones congresales, p. 1 | Dominio público | [Commons](https://commons.wikimedia.org/?curid=82973298) |
| doc_004 | impreso | Folleto del MINSA/DIGESA sobre limpieza y desinfección en domicilios (2020) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=88146070) |
| doc_005 | impreso | Comunicado oficial N.° 21 del MINSA (2020) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=88316494) |
| doc_006 | impreso | Mensaje a la Nación sobre el estado de emergencia (15-03-2020), p. 1 | Dominio público | [Commons](https://commons.wikimedia.org/?curid=88144980) |
| doc_007 | impreso | Boletín 169 de la Liga de Fútbol de Lima (1935) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=142430409) |
| doc_008 | impreso | Acta de creación de los símbolos del distrito de Mache (2023) | CC BY 4.0 | [Commons](https://commons.wikimedia.org/?curid=141457340) |
| doc_009 | foto | Comunicado impreso con el horario de una ruta de transporte (2026) | CC0 | [Commons](https://commons.wikimedia.org/?curid=200440517) |
| doc_010 | foto | Titulares de periódicos en un puesto de venta (2024) | CC0 | [Commons](https://commons.wikimedia.org/?curid=150148728) |
| doc_011 | foto | Portadas de diarios en un quiosco de Lima (2018) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=68497549) |
| doc_012 | foto | Menú impreso de un restaurante de Lima (2012) | CC BY 2.0 | [Commons](https://commons.wikimedia.org/?curid=79662461) |
| doc_013 | foto | Cartel de prevención frente al COVID-19 en la UNMSM (2022) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=120513496) |
| doc_014 | foto | Número 1 del diario El Peruano en vitrina (Museo Bolivariano de Pativilca, 2018) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=73178570) |
| doc_015 | foto | Panel informativo de la Huaca Melgarejo (2022) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=124562195) |
| doc_016 | foto | Afiche informativo sobre la vacunación contra la COVID-19 (2021) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=109659847) |
| doc_017 | manuscrito | Carta de Agustín Gamarra (1822) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155421) |
| doc_018 | manuscrito | Carta de César Vallejo a José María Eguren | Dominio público | [Commons](https://commons.wikimedia.org/?curid=31277483) |
| doc_019 | manuscrito | Carta de Carlos J. Bachmann a Ricardo Palma (1912), p. 1 | CC0 | [Commons](https://commons.wikimedia.org/?curid=196947131) |
| doc_020 | manuscrito | Carta de Miguel Grau (1879) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=110650064) |
| doc_021 | manuscrito | Nicolás de Piérola al prefecto de Lambayeque (1897) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155538) |
| doc_022 | manuscrito | Nota de José Balta (1869) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155518) |
| doc_023 | manuscrito | Nota del general Pedro Bermúdez (1850) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155438) |
| doc_024 | manuscrito | Carta de Miguel San Román a Agustín Gamarra (1840) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155493) |
| doc_025 | manuscrito | Nota de José Rufino Echenique (1843) | Dominio público | [Commons](https://commons.wikimedia.org/?curid=147155487) |

| Grabación | Clips | Descripción | Licencia | Enlace |
|---|---|---|---|---|
| rec_01 | 3 | Entrevista en la calle a una vendedora en El Agustino, Lima (2022) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=119251161) |
| rec_02 | 4 | Entrevista en la calle a dos vendedoras en El Agustino, Lima (2022) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=119251160) |
| rec_03 a rec_06 | 11 | Cuatro entrevistas grabadas por estudiantes de la UNMSM | CC BY-SA 4.0 | [1](https://commons.wikimedia.org/?curid=121363454) · [2](https://commons.wikimedia.org/?curid=121363453) · [3](https://commons.wikimedia.org/?curid=121363455) · [4](https://commons.wikimedia.org/?curid=121363456) |
| rec_07 | 5 | Dirigente de una organización de mujeres indígenas en conferencia de prensa (2022) | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=121219890) |
| rec_08 | 10 | Testimonio de una mujer de Junín | CC BY-SA 4.0 | [Commons](https://commons.wikimedia.org/?curid=149472732) |
| OpenSLR 73 | 120 | Oraciones leídas (`es_pe_female.zip`, `es_pe_male.zip`) | CC BY-SA 4.0 | [OpenSLR](https://www.openslr.org/73/) |

</details>

## 3. Privacidad y seudonimización

**Criterio:**
- **Se reemplazan:** las personas **privadas** (entrevistados, particulares mencionados), y cualquier DNI, celular, correo o RUC de persona natural.
- **Se conservan:** las autoridades que firman en su cargo, los personajes históricos y los dirigentes que hablan en conferencia de prensa.

**Procedimiento:**
1. **Regex** para los identificadores: `[DNI]`, `[TELEFONO]`, `[CORREO]` y `[RUC]`.
2. **NER** `mrm8488/bert-spanish-cased-finetuned-ner`: propone candidatos a nombre de persona en los OCR, los ASR, las salidas de todos los modelos y las referencias manuales.
3. **Revisión humana** de los 146 candidatos: 39 privados, 23 públicos y 84 falsos positivos.
4. **Reemplazo** difuso (`voz.seudonimizar`, rapidfuzz ≥ 85) por `[PERSONA_n]`: hubo 32 reemplazos en documentos y 28 en audio.
5. **Chequeo de fugas** antes de cada push: cada nombre privado se busca en todos los archivos versionados, notebooks incluidos. **Resultado: 0.**

**Qué no se publica:**
- los originales y los audios crudos (`datos/raw/`);
- el diccionario con los nombres reales;
- las imágenes o los clips en los que se ve o se oye el nombre de una persona privada. En total son 14 archivos: el dashboard muestra su texto seudonimizado y sus metadatos, sin imagen ni audio.

Los enlaces de atribución usan `?curid=<pageid>`, así ninguna URL lleva un nombre privado.

## 4. Dos hallazgos

1. **El tipo de documento importa más que el modelo, y cada modelo falla a su manera.**
   - En impresos nítidos, todos los modelos andan bien: CER de 0.06 a 0.15.
   - En **manuscritos**, PaddleOCR se desploma (0.52 en el siglo XX y 0.73 en el XIX) y casi todo su error son **borrados**: no detecta los renglones en cursiva. Gemini, leyendo la imagen entera, baja a 0.05–0.06.
   - En cambio, en las **fotos** Gemini pierde (0.70 frente a 0.43), porque **agrega** texto. En la vitrina de *El Peruano* transcribe la letra pequeña que la persona marcó como ilegible y repite un bloque entero (CER 1.7).

   Un VLM no deja huecos: rellena. Por eso la recomendación es híbrida: PaddleOCR local para los impresos y un VLM solo para los manuscritos.
2. **En audio, lo difícil es la señal, no el tamaño del modelo.**
   - Parakeet queda a 0.014 de WER de turbo y procesa **9 veces más rápido** (RTF 0.27 frente a 2.54, en CPU).
   - Los tres modelos fallan igual en la calle y en las conversaciones de varios hablantes (WER de 0.32 a 0.57), y casi no fallan en lectura limpia (0.00–0.17).
   - **El VAD es imprescindible:** sobre 5 tramos sin voz, `small` sin VAD escribió texto en 4 (tres veces "¡Suscríbete!", aprendido de subtítulos de YouTube). Con VAD, en ninguno.

## 5. Dos limitaciones

1. **Muestra chica y referencias conservadoras.**
   - Con 10 documentos y 10 clips, un solo archivo (doc_014) invierte el resultado de las fotos.
   - Se evaluó el 40 % superior de cada página para que las referencias manuales fueran viables.
   - Cuando la persona marca como `[?]` lo que un modelo sí lee, el modelo queda penalizado.
   - La diarización no tiene verdad de terreno: en rec_02 detectó 1 hablante donde hay 3.
2. **Dependencia de una API y desvío del modelo del corpus.**
   - Gemini 3.8 y 3.7 Flash estaban saturados (503). Se usaron los respaldos de la cadena congelada: 170 de 178 extracciones con `gemini-3.5-flash-lite`, y el modelo queda registrado fila por fila.
   - Usar un VLM implica enviar la imagen a un tercero.
   - Las transcripciones publicadas del corpus son de `small`, que se eligió antes de medir porque turbo no era viable en CPU. La evaluación recomienda Parakeet; retranscribir costaría ~11 min de CPU.

## 6. Resultados

| Modalidad | Modelo | Rol | CER / WER agregado | Costo |
|---|---|---|---|---|
| OCR | PaddleOCR 3.7 PP-OCRv6, por defecto | Línea base | 0.310 | Local, ~8 s por recorte |
| OCR | PaddleOCR sin corrección geométrica | Ablación | 0.323 | Local, ~7 s por recorte |
| OCR | **Gemini Flash (VLM)** | **Final** | **0.138** | API, ~23 s por recorte |
| ASR | faster-whisper `small` | Línea base | 0.224 | Local, RTF 1.02 |
| ASR | faster-whisper large-v3-turbo | Más grande | 0.125 | Local, RTF 2.54 |
| ASR | **Parakeet TDT 0.6B v3** | **Final** | **0.139** | Local, RTF 0.27 |

**Regla de decisión,** congelada antes de medir: menor error agregado y, si la diferencia es menor que 0.02, el modelo más barato.

**LLM:**
- Auditoría humana de 20 extracciones: **11/20 completamente correctas**. Las citas y el número de hablantes salieron siempre bien.
- Los errores se concentran en los manuscritos del siglo XIX, donde el LLM "corrige" el OCR o infiere datos.
- El chequeo automático de literalidad da 96 %: es más optimista que la auditoría humana.

## 7. Alcance y bonus

- **Hecho:**
  - **NER peruano:** entidades por modalidad en `datos/ner_entidades.csv`, con un gráfico en el notebook 02.
  - **"Quién dijo qué":** citas por hablante en los clips de varias voces, de forma **cualitativa**, porque validarlas contra la misma diarización sería circular.
  - **Ablación** de la corrección geométrica de PaddleOCR.
- **No hecho, y por qué:**
  - **ASR en quechua o aimara:** no hay audios con transcripción humana para medir un WER honesto.
  - **Voxtral:** no hay GPU.
  - **Procesamiento en vivo en el dashboard:** los modelos no caben en la memoria de Community Cloud.
  - **Video:** falta de tiempo.

## 8. Estructura y bitácora

```text
tarea_2/
├── 01_extraccion.ipynb       # Parte 1: corpus, OCR, ASR, seudonimización, EDA y exportación
├── 02_evaluacion.ipynb       # Parte 2: CER/WER, LLM, auditoría, casos difíciles, decisión y bonus
├── app.py                    # Parte 3: dashboard (solo lee datos/ y modelos/)
├── voz.py                    # copia de la sesión 13 del curso (MIT) + clave faster-whisper-small
├── ocr.py, corpus.py, privacidad.py, extraccion.py, paleta.py
├── herramientas/estacion_manual.py   # app local para referencias ciegas, privacidad y auditoría
├── requirements.txt          # solo el dashboard
├── requirements-notebooks.txt
├── evidencias_ia/            # capturas numeradas de la conversación con la IA
├── datos/                    # CSV de la entrega, salidas_asr/ (SRT y TXT), ejemplos/ (miniaturas y MP3 publicables)
└── modelos/                  # configuraciones.json (congelado antes de medir) y metricas.json
```

**Bitácora:**
- Issues [#12](https://github.com/rodnm/intermediate-python-hw/issues/12), [#13](https://github.com/rodnm/intermediate-python-hw/issues/13) y [#14](https://github.com/rodnm/intermediate-python-hw/issues/14), uno por parte, cerrados por su PR.
- La muestra y las configuraciones se commitearon **antes** que cualquier métrica: lo muestra el `git log` impreso al inicio del notebook 02.

**Evidencias de IA:** `evidencias_ia/` contiene capturas numeradas de la conversación con Claude Code. Son un render fiel de las transcripciones de la sesión, con claves, correos y nombres privados redactados antes de capturar. La IA escribió código y textos. Las referencias manuales, la revisión de privacidad y la auditoría las hizo una persona, en la estación local, sin ver las salidas de los modelos.
