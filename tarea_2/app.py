"""Dashboard de la Tarea 2: Voces y documentos peruanos, del archivo escaneado al texto buscable.

Solo lee los resultados ya calculados en los notebooks (datos/*.csv y modelos/metricas.json): no corre OCR, ASR ni LLM.
Ejecutar desde la raíz del repositorio para cargar el tema:  streamlit run tarea_2/app.py
"""
import difflib
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import paleta

RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"

st.set_page_config(page_title="Voces y documentos peruanos", page_icon="📜", layout="wide")


@st.cache_data
def leer(nombre: str) -> pd.DataFrame:
    ruta = DATOS / nombre
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


@st.cache_data
def metricas() -> dict:
    ruta = RAIZ / "modelos" / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


def plano(texto: str) -> str:
    """Minúsculas y sin tildes, para buscar sin importar cómo se escribió."""
    t = unicodedata.normalize("NFD", str(texto or "").lower())
    return "".join(ch for ch in t if unicodedata.category(ch) != "Mn")


def diff_html(referencia: str, hipotesis: str) -> str:
    """Diferencias palabra a palabra: tachado = falta en el modelo, resaltado = el modelo lo agregó o lo cambió."""
    a, b = referencia.split(), hipotesis.split()
    partes = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            partes.append(html.escape(" ".join(a[i1:i2])))
        if op in ("delete", "replace"):
            partes.append(f"<del style='color:{paleta.CONTRASTE}'>{html.escape(' '.join(a[i1:i2]))}</del>")
        if op in ("insert", "replace"):
            partes.append(f"<mark style='background:{paleta.FONDO_ALTO};color:{paleta.TEXTO}'>{html.escape(' '.join(b[j1:j2]))}</mark>")
    return " ".join(partes)


docs, trans = leer("documentos.csv"), leer("transcripciones.csv")
ext, fuentes = leer("extraccion_llm.csv"), leer("fuentes.csv")
p_ocr, p_asr, det = leer("predicciones_ocr.csv"), leer("predicciones_asr.csv"), leer("metricas_por_archivo.csv")
segs = leer("segmentos_asr.csv")
m = metricas()
if docs.empty or trans.empty:
    st.error("Faltan datos/documentos.csv o datos/transcripciones.csv: ejecutar los notebooks 01 y 02.")
    st.stop()
campos = {i: json.loads(c) for i, c in zip(ext.get("id", []), ext.get("campos_json", []))}

# ── encabezado
st.title("¿Pueden convertirse documentos escaneados y entrevistas en audio en texto estructurado y buscable?")
fin = m.get("final", {})
if fin:
    st.markdown(f"**Sí, con matices.** Sobre una muestra manual de 10 documentos y 10 clips, el mejor OCR ({fin['ocr']}) "
                f"erra **{fin['cer']:.0%} de los caracteres** y el mejor ASR ({fin['asr']}) **{fin['wer']:.0%} de las "
                f"palabras**. Los manuscritos siguen siendo el punto débil.")
with st.expander("Cómo se hizo"):
    st.markdown(
        "- **Corpus:** 25 documentos peruanos de Wikimedia Commons (8 impresos, 8 fotos, 9 manuscritos) y "
        f"{m.get('minutos_audio', '…')} min de audio en español peruano (entrevistas de Commons y OpenSLR 73), "
        "todo con licencia abierta.\n"
        "- **OCR:** PaddleOCR 3.7 (PP-OCRv6), comparado con Gemini Flash como modelo de visión y lenguaje.\n"
        "- **ASR:** faster-whisper con VAD y diarización (`voz.py` de la sesión 13), comparado con Parakeet TDT v3.\n"
        "- **Evaluación:** CER/WER contra transcripciones manuales ciegas, con la muestra y los parámetros congelados "
        "antes de medir.\n"
        "- **LLM:** Gemini con esquema JSON para campos estructurados, auditado a mano en 20 casos.\n"
        "- **Privacidad:** las personas privadas están seudonimizadas como [PERSONA_n].")

# ── filtros
with st.sidebar:
    st.header("Filtros")
    q = st.text_input("Buscar en los textos OCR y ASR", placeholder="p. ej. emergencia, mercado, Lima")
    tipos = st.multiselect("Tipo de documento", ["impreso", "foto", "manuscrito"], default=["impreso", "foto", "manuscrito"])
    modelos_ocr = sorted(p_ocr.modelo.unique()) if not p_ocr.empty else []
    modelos_asr = sorted(p_asr.modelo.unique()) if not p_asr.empty else []
    sel_ocr = st.multiselect("Modelos OCR", modelos_ocr, default=modelos_ocr)
    sel_asr = st.multiselect("Modelos ASR", modelos_asr, default=modelos_asr)
    fuentes_audio = sorted(trans.fuente.unique())
    sel_fuente = st.multiselect("Fuente de audio", fuentes_audio, default=fuentes_audio)
    dmax = float(trans.duracion_seg.max())
    rango = st.slider("Duración del clip (s)", 0.0, dmax, (0.0, dmax))

f_docs = docs[docs.tipo.isin(tipos)]
f_trans = trans[trans.fuente.isin(sel_fuente) & trans.duracion_seg.between(*rango)]
if q:
    f_docs = f_docs[f_docs.texto_ocr.map(plano).str.contains(plano(q), regex=False)]
    f_trans = f_trans[f_trans.texto_asr.map(plano).str.contains(plano(q), regex=False)]

tab_res, tab_doc, tab_aud, tab_cmp, tab_eti = st.tabs(["Resumen", "Documentos", "Audios", "Comparativa de modelos", "Fuentes y ética"])

with tab_res:
    c1, c2, c3, c4, c5 = st.columns(5)
    base = m.get("linea_base", {})
    c1.metric("Documentos procesados", len(docs))
    c2.metric("Minutos de audio transcritos", f"{trans.duracion_seg.sum() / 60:.1f}")
    if fin:
        c3.metric(f"CER ({fin['ocr']})", f"{fin['cer']:.2f}", f"{fin['cer'] - base.get('cer', fin['cer']):+.2f} vs. base",
                  delta_color="inverse")
        c4.metric(f"WER ({fin['asr']})", f"{fin['wer']:.2f}", f"{fin['wer'] - base.get('wer', fin['wer']):+.2f} vs. base",
                  delta_color="inverse")
    acu = m.get("acuerdo_llm_humano", {})
    if acu:
        c5.metric("Acuerdo LLM vs. auditoría", f"{acu['correctas'] / acu['total']:.0%}", f"{acu['correctas']}/{acu['total']} casos",
                  delta_color="off")
    st.caption(f"Con los filtros actuales: {len(f_docs)} documentos y {len(f_trans)} clips.")
    g1, g2 = st.columns(2)
    cuenta = f_docs.tipo.value_counts().reset_index()
    g1.plotly_chart(px.bar(cuenta, x="tipo", y="count", color="tipo", color_discrete_map=paleta.COLORES_TIPO, template="paleta",
                           labels={"count": "Documentos", "tipo": "Tipo"}, title="Documentos por tipo"), width="stretch")
    g2.plotly_chart(px.histogram(f_trans, x="duracion_seg", color="fuente", nbins=30, template="paleta",
                                 color_discrete_map=paleta.COLORES_FUENTE, labels={"duracion_seg": "Duración (s)"},
                                 title="Duración de los clips"), width="stretch")

with tab_doc:
    vista = f_docs[["id", "tipo", "fuente", "texto_ocr", "texto_referencia"]]
    st.dataframe(vista, width="stretch", hide_index=True)
    st.download_button("Descargar CSV (textos seudonimizados)", vista.to_csv(index=False).encode("utf-8"),
                       "documentos_filtrados.csv", "text/csv")
    st.subheader("Explorador de documentos")
    if f_docs.empty:
        st.info("Ningún documento coincide con los filtros.")
    else:
        i = st.selectbox("Documento", f_docs.id, format_func=lambda x: f"{x} · {docs.set_index('id').tipo[x]} · "
                         f"{fuentes.set_index('id').titulo.get(x, '')}")
        d = docs.set_index("id").loc[i]
        izq, der = st.columns([2, 3])
        img = DATOS / "ejemplos" / f"{i}.jpg"
        if bool(d.get("imagen_publicable", False)) and img.exists():
            f = fuentes.set_index("id").loc[i]
            izq.image(str(img), caption=f"{f.titulo} · {f.autor} · {f.licencia} · {f.fuente_url}")
        else:
            izq.info("La imagen no se publica (licencia o datos personales); se muestran el texto y los metadatos.")
        der.markdown("**Texto OCR (PaddleOCR, página completa)**")
        der.text_area("ocr", d.texto_ocr if isinstance(d.texto_ocr, str) else "", height=220, disabled=True,
                      label_visibility="collapsed")
        der.markdown("**Campos estructurados (LLM)**")
        der.json(campos.get(i, {}))
        if isinstance(d.texto_referencia, str) and d.texto_referencia.strip():
            st.markdown("**Muestra de evaluación:** referencia manual del 40 % superior de la página y CER de cada modelo")
            cols = st.columns(max(1, len(sel_ocr)))
            for col, mod in zip(cols, sel_ocr):
                v = det[(det.id == i) & (det.modelo == mod)].valor
                h = p_ocr[(p_ocr.id == i) & (p_ocr.modelo == mod)].texto
                col.metric(mod, f"{v.iloc[0]:.2f}" if len(v) else "–")
                if len(h):
                    col.markdown(f"<div style='font-size:0.85em'>{diff_html(d.texto_referencia, str(h.iloc[0]))}</div>",
                                 unsafe_allow_html=True)
            with st.expander("Referencia manual"):
                st.text(d.texto_referencia)

with tab_aud:
    vista = f_trans[["id", "fuente", "duracion_seg", "n_hablantes", "texto_asr", "texto_referencia"]]
    st.dataframe(vista, width="stretch", hide_index=True)
    st.download_button("Descargar CSV (textos seudonimizados)", vista.to_csv(index=False).encode("utf-8"),
                       "transcripciones_filtradas.csv", "text/csv")
    st.subheader("Explorador de audios")
    if f_trans.empty:
        st.info("Ningún clip coincide con los filtros.")
    else:
        i = st.selectbox("Clip", f_trans.id, format_func=lambda x: f"{x} · {trans.set_index('id').fuente[x]} · "
                         f"{trans.set_index('id').duracion_seg[x]:.0f} s")
        a = trans.set_index("id").loc[i]
        mp3 = DATOS / "ejemplos" / f"{i}.mp3"
        if bool(a.get("audio_publicable", False)) and mp3.exists():
            st.audio(str(mp3), format="audio/mpeg")
            f = fuentes.set_index("id").loc[i]
            st.caption(f"{f.titulo} · {f.autor} · {f.licencia} · {f.fuente_url}")
        else:
            st.info("Este audio no se reproduce aquí (sin ejemplo publicable); se muestran la transcripción y los metadatos.")
        st.markdown(f"**Transcripción** · {int(a.n_hablantes)} hablante(s) detectado(s) por diarización")
        # negrita solo en las marcas de hablante: [PERSONA_n] y [?] quedan tal cual
        st.markdown(re.sub(r"\[HABLANTE_(\d+)\]", r"\n\n**[HABLANTE_\1]**", str(a.texto_asr)))
        cit = campos.get(i, {}).get("citas_clave", [])
        if cit:
            st.markdown("**Quién dijo qué** (citas del LLM; el hablante viene de la diarización)")
            st.table(pd.DataFrame(cit))
        with st.expander("Campos estructurados (LLM)"):
            st.json(campos.get(i, {}))
        if isinstance(a.texto_referencia, str) and a.texto_referencia.strip():
            st.markdown("**Muestra de evaluación:** referencia manual y WER de cada modelo (mismos tramos de voz)")
            cols = st.columns(max(1, len(sel_asr)))
            for col, mod in zip(cols, sel_asr):
                v = det[(det.id == i) & (det.modelo == mod)].valor
                h = p_asr[(p_asr.id == i) & (p_asr.modelo == mod)].texto
                col.metric(mod, f"{v.iloc[0]:.2f}" if len(v) else "–")
                if len(h):
                    col.markdown(f"<div style='font-size:0.85em'>{diff_html(a.texto_referencia, str(h.iloc[0]))}</div>",
                                 unsafe_allow_html=True)

with tab_cmp:
    if det.empty:
        st.info("Las métricas se generan en el notebook 02.")
    else:
        d_ocr = det[det.modelo.isin(sel_ocr)]
        d_asr = det[det.modelo.isin(sel_asr)]
        st.plotly_chart(px.bar(d_ocr.groupby(["estrato", "modelo"], as_index=False).valor.mean(), x="estrato", y="valor",
                               color="modelo", barmode="group", color_discrete_map=paleta.COLORES_MODELO, template="paleta",
                               labels={"valor": "CER medio", "estrato": "Tipo de documento"},
                               title="CER por tipo de documento y modelo OCR"), width="stretch")
        wer_rtf = pd.DataFrame({"modelo": list(m.get("wer", {})), "wer": list(m.get("wer", {}).values()),
                                "rtf": [m.get("rtf", {}).get(k) for k in m.get("wer", {})]})
        wer_rtf = wer_rtf[wer_rtf.modelo.isin(sel_asr)]
        st.plotly_chart(px.scatter(wer_rtf, x="rtf", y="wer", text="modelo", color="modelo", template="paleta",
                                   color_discrete_map=paleta.COLORES_MODELO,
                                   labels={"rtf": "Factor de tiempo real", "wer": "WER agregado"},
                                   title="Precisión contra costo de los modelos ASR").update_traces(
            textposition="top center", marker_size=14), width="stretch")
        tabla = pd.DataFrame({"modelo": list(m.get("cer", {})) + list(m.get("wer", {})),
                              "métrica": ["CER"] * len(m.get("cer", {})) + ["WER"] * len(m.get("wer", {})),
                              "agregado": list(m.get("cer", {}).values()) + list(m.get("wer", {}).values())})
        st.dataframe(tabla, hide_index=True, width="stretch")

with tab_eti:
    st.markdown(
        "**Seudonimización.** Las personas privadas (por ejemplo, entrevistados en la calle) se reemplazan por "
        "`[PERSONA_n]`, y los DNI, teléfonos y correos, por marcas. Se conservan las autoridades que firman en su cargo, "
        "los personajes históricos y los dirigentes en conferencia de prensa. Los candidatos los propuso un modelo NER y "
        "los clasificó una persona. Las imágenes y los audios se publican solo si su licencia lo permite y no exponen a "
        "una persona privada.")
    st.markdown("**Fuentes y licencias** (el enlace `?curid=` lleva a la página del archivo en Commons)")
    st.dataframe(fuentes, width="stretch", hide_index=True)
