"""Dashboard de la Tarea 1: clasificador de titulares peruanos (La República + El Comercio).

Ejecutar desde la carpeta tarea_1/:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
DATOS, MODELO = BASE / "datos", BASE / "modelo"
MODELO_HUB = "rodnm/distilbert-noticias-peru"     # respaldo cuando no hay pesos locales (p. ej. en la nube)
MODELO_EMB = "intfloat/multilingual-e5-small"
CATEGORIAS = ["politica", "economia", "deportes", "ciencia_tecnologia"]
NOMBRES = {"politica": "Política", "economia": "Economía", "deportes": "Deportes",
           "ciencia_tecnologia": "Ciencia y tecnología"}
COLORES_MEDIO = {"El Comercio": "#1f77b4", "La Republica": "#d62728"}

st.set_page_config(page_title="Clasificador de titulares peruanos", page_icon="🗞️", layout="wide")


@st.cache_data
def cargar_noticias():
    return pd.read_csv(DATOS / "noticias.csv")


@st.cache_data
def cargar_metricas():
    return json.loads((MODELO / "metricas.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_opcional(nombre):
    """Archivos generados por 03_innovacion.ipynb; si no existen, la pestaña lo avisa."""
    ruta = DATOS / nombre
    if not ruta.exists():
        return None
    return np.load(ruta) if ruta.suffix == ".npy" else pd.read_csv(ruta)


@st.cache_resource(show_spinner="Cargando el clasificador...")
def cargar_clasificador():
    from transformers import pipeline

    origen = str(MODELO) if (MODELO / "model.safetensors").exists() else MODELO_HUB
    return pipeline("text-classification", model=origen, top_k=None, device="cpu")


@st.cache_resource(show_spinner="Cargando el modelo de embeddings...")
def cargar_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELO_EMB, device="cpu")


def clasificar(titular):
    salida = cargar_clasificador()(titular)
    salida = salida[0] if isinstance(salida[0], list) else salida
    probs = pd.DataFrame(salida).rename(columns={"label": "categoria", "score": "probabilidad"})
    return probs.sort_values("probabilidad", ascending=False).reset_index(drop=True)


noticias = cargar_noticias()
metricas = cargar_metricas()

# --- Encabezado ---------------------------------------------------------------------------------------------
st.title("🗞️ ¿Pueden convertir titulares peruanos en un clasificador de noticias?")
st.markdown(
    f"**{len(noticias)} titulares reales** de La República y El Comercio, publicados entre el "
    f"{pd.to_datetime(noticias['fecha']).min():%d/%m} y el {pd.to_datetime(noticias['fecha']).max():%d/%m/%Y}, "
    f"clasificados en 4 categorías por un DistilBERT multilingüe ajustado con esos mismos titulares."
)
with st.expander("Metodología en 4 pasos"):
    st.markdown(f"""
1. **Recolección respetuosa.** El Comercio sale de sus sitemaps por sección. La República solo publica ~48 h en
   RSS y su `robots.txt` prohíbe paginar, así que esas fuentes se usaron como semillas de un rastreo de enlaces
   relacionados, con pausas de 1–2 s y validación de `robots.txt`.
2. **Etiquetas.** La etiqueta de referencia es la sección donde el diario publicó la nota. Además, un LLM
   (`{metricas['modelo_llm']}`) etiquetó todo el dataset (supervisión débil), con una auditoría humana a ciegas
   de 20 casos.
3. **Modelos.** Se compararon un zero-shot (BETO-XNLI), el LLM y un DistilBERT ajustado 3 épocas, sobre un test
   estratificado que no se usó para entrenar ni para elegir el modelo.
4. **Resultado.** El modelo ajustado obtiene un F1 macro de **{metricas['f1_macro_test']:.3f}** en test, contra
   **{metricas['f1_macro_zero_shot']:.3f}** del zero-shot.
""")

# --- Filtros ------------------------------------------------------------------------------------------------
st.sidebar.header("Filtros")
medios = st.sidebar.multiselect("Medio", sorted(noticias["medio"].unique()), default=sorted(noticias["medio"].unique()))
categorias = st.sidebar.multiselect("Categoría", CATEGORIAS, default=CATEGORIAS, format_func=NOMBRES.get)
mascara = noticias["medio"].isin(medios) & noticias["etiqueta"].isin(categorias)
filtradas = noticias[mascara]
st.sidebar.caption(f"{len(filtradas)} de {len(noticias)} noticias con los filtros actuales.")
st.sidebar.caption("Los filtros afectan el gráfico, la tabla, el buscador y las entidades.")

# --- Indicadores --------------------------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Noticias recolectadas", len(noticias), delta=f"{len(filtradas)} con los filtros", delta_color="off")
col2.metric("F1 macro del modelo en test", f"{metricas['f1_macro_test']:.3f}",
            delta=f"{metricas['f1_macro_test'] - metricas['f1_macro_zero_shot']:+.3f} vs zero-shot")
col3.metric("Acuerdo LLM vs auditoría humana", f"{metricas['acuerdo_llm_humano'] / metricas['n_auditoria']:.0%}",
            delta=f"{metricas['acuerdo_llm_humano']}/{metricas['n_auditoria']} casos", delta_color="off")

tab_probador, tab_datos, tab_buscador, tab_sesgo, tab_ner = st.tabs(
    ["🔮 Probador en vivo", "📊 Explorar datos", "🔎 Buscador semántico", "⚖️ Sesgo LR vs EC", "🏷️ Entidades (NER)"])

# --- Probador en vivo ---------------------------------------------------------------------------------------
with tab_probador:
    st.subheader("Escribe o pega un titular nuevo")
    titular = st.text_input("Titular", placeholder="Ej.: BCR mantiene su tasa de referencia en 4,25 %")
    if st.button("Clasificar", type="primary", disabled=not titular.strip()):
        probs = clasificar(titular.strip())
        mejor = probs.iloc[0]
        st.success(f"Categoría predicha: **{NOMBRES[mejor['categoria']]}** · probabilidad {mejor['probabilidad']:.1%}")
        if mejor["probabilidad"] < 0.6:
            st.warning("Confianza baja: el titular mezcla temas o se aleja de lo que el modelo vio al entrenar.")
        fig = px.bar(probs, x="probabilidad", y=probs["categoria"].map(NOMBRES), orientation="h", range_x=[0, 1],
                     text=probs["probabilidad"].map("{:.1%}".format),
                     labels={"probabilidad": "Probabilidad", "y": ""}, title="Probabilidad por categoría")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")

# --- Explorar datos -----------------------------------------------------------------------------------------
with tab_datos:
    conteo = filtradas.groupby(["etiqueta", "medio"]).size().rename("noticias").reset_index()
    conteo["categoria"] = conteo["etiqueta"].map(NOMBRES)
    fig = px.bar(conteo, x="categoria", y="noticias", color="medio", barmode="group", text="noticias",
                 color_discrete_map=COLORES_MEDIO, labels={"categoria": "Categoría", "medio": "Medio"},
                 category_orders={"categoria": [NOMBRES[c] for c in CATEGORIAS]},
                 title="Distribución de categorías por medio (con los filtros aplicados)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(filtradas, hide_index=True, width="stretch",
                 column_config={"url": st.column_config.LinkColumn("url", display_text="abrir nota")})
    st.download_button("⬇️ Descargar CSV filtrado", filtradas.to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

# --- Buscador semántico -------------------------------------------------------------------------------------
with tab_buscador:
    embeddings = cargar_opcional("embeddings.npy")
    if embeddings is None:
        st.info("Ejecuta `03_innovacion.ipynb` para generar `datos/embeddings.npy`.")
    else:
        st.subheader("Noticias más parecidas a una consulta")
        st.caption("Búsqueda por significado (embeddings e5 + similitud coseno), no por palabras exactas.")
        consulta = st.text_input("Consulta", placeholder="Ej.: sube el dólar")
        k = st.slider("Cantidad de resultados", 3, 15, 5)
        if consulta.strip():
            q = cargar_embedder().encode([f"query: {consulta.strip()}"], normalize_embeddings=True)[0]
            sims = embeddings @ q
            orden = [i for i in np.argsort(-sims) if mascara.iloc[i]][:k]
            resultados = noticias.iloc[orden][["medio", "etiqueta", "titular", "url"]].assign(similitud=sims[orden])
            st.dataframe(resultados, hide_index=True, width="stretch", column_config={
                "url": st.column_config.LinkColumn("url", display_text="abrir nota"),
                "similitud": st.column_config.ProgressColumn("similitud", min_value=0.0, max_value=1.0, format="%.3f"),
            })

# --- Sesgo LR vs EC -----------------------------------------------------------------------------------------
with tab_sesgo:
    pares = cargar_opcional("pares_sesgo.csv")
    if pares is None:
        st.info("Ejecuta `03_innovacion.ipynb` para generar `datos/pares_sesgo.csv`.")
    else:
        st.subheader("La misma noticia contada por los dos diarios")
        st.caption("Pares emparejados por similitud de embeddings (±2 días) y confirmados a mano.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Pares confirmados", len(pares))
        c2.metric("Misma sección editorial", f"{(pares['seccion_lr'] == pares['seccion_ec']).mean():.0%}")
        c3.metric("Misma predicción del modelo", f"{(pares['pred_lr'] == pares['pred_ec']).mean():.0%}")
        solo_cambios = st.toggle("Mostrar solo los pares donde cambia la sección o la predicción")
        vista = pares[(pares["pred_lr"] != pares["pred_ec"]) | (pares["seccion_lr"] != pares["seccion_ec"])] \
            if solo_cambios else pares
        st.dataframe(vista[["titular_lr", "titular_ec", "seccion_lr", "seccion_ec", "pred_lr", "prob_lr",
                            "pred_ec", "prob_ec"]], hide_index=True, width="stretch")

# --- Entidades (NER) ----------------------------------------------------------------------------------------
with tab_ner:
    entidades = cargar_opcional("ner_entidades.csv")
    if entidades is None:
        st.info("Ejecuta `03_innovacion.ipynb` para generar `datos/ner_entidades.csv`.")
    else:
        tipos = {"PER": "Personas", "ORG": "Organizaciones", "LOC": "Lugares", "MISC": "Otros"}
        tipo = st.radio("Tipo de entidad", list(tipos), format_func=tipos.get, horizontal=True)
        sel = entidades[(entidades["tipo"] == tipo) & entidades["url"].isin(filtradas["url"])]
        top = sel.groupby(["entidad", "medio"]).size().rename("noticias").reset_index()
        orden = top.groupby("entidad")["noticias"].sum().nlargest(15).index
        fig = px.bar(top[top["entidad"].isin(orden)], x="noticias", y="entidad", color="medio", orientation="h",
                     color_discrete_map=COLORES_MEDIO, height=520, labels={"entidad": "", "medio": "Medio"},
                     title=f"{tipos[tipo]} más mencionadas (noticias en que aparecen)")
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")

st.divider()
st.caption("Fuentes: larepublica.pe y elcomercio.pe (RSS, sitemaps y artículos públicos, respetando robots.txt). "
           "Proyecto académico — Python Intermedio, 2026-II.")
