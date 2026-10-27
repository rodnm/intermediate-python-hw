"""Estación de trabajo manual (app LOCAL, no se despliega).

Tres modos:
- Referencias: transcripción ciega de los 10 recortes y los 10 clips de la muestra. Nunca muestra salidas de modelos.
- Privacidad: clasifica los candidatos a nombre propio (privado / público / no es persona) y marca qué imágenes
  y audios se pueden publicar.
- Auditoría: juzga campo por campo las 20 extracciones del LLM sorteadas.

Ejecutar desde la raíz del repositorio:
    tarea_2/.venv/Scripts/streamlit run tarea_2/herramientas/estacion_manual.py --server.port 8502
"""
import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

T2 = Path(__file__).resolve().parents[1]
RAW = T2 / "datos" / "raw"
REF = RAW / "referencias"
REF.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Estación manual · Tarea 2", layout="wide")
modo = st.sidebar.radio("Modo", ["Referencias", "Privacidad", "Auditoría"])


def reproductor(ruta: Path):
    """Audio con velocidad 0.75x/1x y saltos de ±5 s (el st.audio estándar no trae velocidad)."""
    b64 = base64.b64encode(ruta.read_bytes()).decode()
    components.html(f"""
    <audio id="a" controls style="width:100%" src="data:audio/wav;base64,{b64}"></audio>
    <div style="margin-top:6px;font-family:sans-serif">
      <button onclick="a.currentTime=Math.max(0,a.currentTime-5)">« 5 s</button>
      <button onclick="a.currentTime=a.currentTime+5">5 s »</button>
      <button onclick="a.playbackRate=0.75">0.75×</button>
      <button onclick="a.playbackRate=1">1×</button>
    </div>""", height=90)


if modo == "Referencias":
    st.title("Referencias manuales (ciegas)")
    st.caption("Transcribí solo lo que ves u oís. Esta pantalla nunca muestra la salida de un modelo. "
               "Guía completa: sección 9.2 del plan.")
    muestra = pd.read_csv(T2 / "datos" / "muestra_eval.csv")
    hechos = {p.stem for p in REF.glob("*.txt")}
    st.progress(len(hechos & set(muestra["id"])) / len(muestra),
                text=f"{len(hechos & set(muestra['id']))} de {len(muestra)} referencias guardadas")
    etiquetas = [f"{'✅' if i in hechos else '⬜'} {i} · {m} · {e}" for i, m, e in muestra[["id", "modalidad", "estrato"]].itertuples(index=False)]
    elegido = st.selectbox("Caso", range(len(muestra)), format_func=lambda k: etiquetas[k])
    fila = muestra.iloc[elegido]
    destino = REF / f"{fila['id']}.txt"
    previo = destino.read_text(encoding="utf-8") if destino.exists() else ""
    izq, der = st.columns([3, 2])
    with izq:
        if fila["modalidad"] == "documento":
            st.image(str(RAW / "eval" / f"{fila['id']}.png"), caption="Región de evaluación: 40 % superior de la página")
            with st.expander("Reglas para documentos"):
                st.markdown("- Todo lo legible del recorte, en orden de lectura (columnas: primero la izquierda).\n"
                            "- Literal: ortografía, tildes, mayúsculas y abreviaturas tal como aparecen.\n"
                            "- Una línea del documento = una línea tuya; la palabra cortada con guion, como aparece.\n"
                            "- Línea cortada por el borde: solo lo que se lee. Palabra ilegible: `[?]` (poco).\n"
                            "- Firma ilegible: se omite. Sello impreso legible: se transcribe.")
        else:
            reproductor(RAW / "audio" / "clips" / f"{fila['id']}.wav")
            with st.expander("Reglas para audio"):
                st.markdown("- Las palabras como se oyen, repeticiones incluidas; muletillas (eh, em, mm) opcionales.\n"
                            "- Números en dígitos o en palabras (se normalizan igual). Ininteligible: `[?]`.\n"
                            "- Nombres de personas: escribilos como se oyen (esto queda local y se seudonimiza).\n"
                            "- No hace falta marcar hablantes.")
    with der:
        texto = st.text_area("Transcripción", value=previo, height=420, key=f"t_{fila['id']}")
        if st.button("Guardar", type="primary"):
            destino.write_text(texto.strip() + "\n", encoding="utf-8")
            st.success(f"Guardado {destino.name}")
            st.rerun()
    # Control de calidad de OpenSLR: solo cuando las 4 referencias de lectura ya están guardadas
    slr = muestra[muestra["estrato"].str.startswith("openslr")]
    if len(slr) and set(slr["id"]) <= hechos:
        with st.expander("Control de calidad (OpenSLR): tu texto contra la frase leída del corpus"):
            import sys
            sys.path.insert(0, str(T2))
            from ocr import wer as wer_normalizado
            corpus_txt = pd.read_csv(RAW / "audio" / "clips_openslr.csv").set_index("id")["texto_corpus"]
            for i in slr["id"]:
                tuyo = (REF / f"{i}.txt").read_text(encoding="utf-8").strip()
                # misma normalización que la métrica: la puntuación y las mayúsculas no cuentan
                wer = wer_normalizado([corpus_txt[i]], [tuyo])
                st.write(f"**{i}** · diferencia {wer:.0%}" + (" ← volvé a escucharlo" if wer > 0.05 else ""))

elif modo == "Privacidad":
    st.title("Revisión de privacidad")
    ruta_c = RAW / "privacidad" / "candidatos.csv"
    if not ruta_c.exists():
        st.info("Todavía no hay candidatos: se generan después del OCR/ASR.")
        st.stop()
    cand = pd.read_csv(ruta_c)
    ruta_r = RAW / "privacidad" / "revision.csv"
    previa = pd.read_csv(ruta_r).set_index("candidato")["clase"].to_dict() if ruta_r.exists() else {}
    st.subheader("1. Candidatos a nombre de persona")
    st.caption("privado = se reemplaza por [PERSONA_n] · publico = autoridad en su cargo, figura histórica, firmante "
               "oficial o dirigente en conferencia · no_es_persona = falso positivo del NER")
    opciones = ["privado", "publico", "no_es_persona"]
    clases = {}
    cand["en_audio"] = cand["ids"].str.contains("clip_")
    st.markdown(f"**A. Candidatos que aparecen en audio ({int(cand.en_audio.sum())})** · aquí están los entrevistados: revisalos uno por uno")
    contenedor_docs = None
    for k, f in cand.sort_values(["en_audio", "apariciones"], ascending=[False, False]).iterrows():
        if not f["en_audio"] and contenedor_docs is None:
            contenedor_docs = st.expander(f"B. Candidatos solo en documentos ({int((~cand.en_audio).sum())}) · sugerencia "
                                          "'publico' (autoridades, personajes históricos o ruido del OCR): revisá rápido")
        zona = contenedor_docs if contenedor_docs is not None else st
        c1, c2 = zona.columns([2, 3])
        c1.markdown(f"**{f['candidato']}** · {f['apariciones']} apariciones")
        c1.caption(f"{f['ids']}")
        if isinstance(f.get("contexto"), str) and f["contexto"]:
            c1.caption(f"… {f['contexto']} …")
        sugerida = previa.get(f["candidato"], f.get("sugerencia", "privado"))
        clases[f["candidato"]] = c2.radio("Clase", opciones, index=opciones.index(sugerida) if sugerida in opciones else 0,
                                          key=f"c{k}", horizontal=True, label_visibility="collapsed")
    st.subheader("2. Imágenes y audios publicables")
    ruta_p = RAW / "privacidad" / "publicables.csv"
    pub = pd.read_csv(ruta_p) if ruta_p.exists() else pd.read_csv(RAW / "privacidad" / "publicables_base.csv")
    st.caption("Desmarcá lo que muestre a una persona privada identificable o datos personales, y los clips donde "
               "se oiga el nombre de una persona privada.")
    pub = st.data_editor(pub, disabled=[c for c in pub.columns if c != "publicable"], hide_index=True,
                         width="stretch", key="pub")
    if st.button("Guardar revisión", type="primary"):
        pd.DataFrame({"candidato": list(clases), "clase": list(clases.values())}).to_csv(ruta_r, index=False)
        pub.to_csv(ruta_p, index=False)
        st.success("Revisión guardada")

else:
    st.title("Auditoría de 20 extracciones del LLM")
    ruta_casos = RAW / "auditoria" / "casos.csv"
    if not ruta_casos.exists():
        st.info("Todavía no hay casos: se generan después de la extracción con el LLM.")
        st.stop()
    casos = pd.read_csv(ruta_casos)
    salida = T2 / "datos" / "auditoria_llm.csv"
    hecho = pd.read_csv(salida) if salida.exists() else pd.DataFrame(columns=["id", "modalidad", "campo", "correcto", "nota"])
    listos = set(hecho["id"])
    st.progress(len(listos & set(casos["id"])) / len(casos), text=f"{len(listos & set(casos['id']))} de {len(casos)} casos auditados")
    with st.expander("Criterios (fijados antes de empezar)", expanded=False):
        st.markdown(
            "- **tema**: correcto si nombra el asunto principal.\n"
            "- **resumen**: correcto si todo lo que afirma está en el texto y no omite el asunto principal.\n"
            "- **campos factuales** (tipo_documento, emisor, destinatario, fecha, lugar, monto): correcto si coincide "
            "con el texto (formatos equivalentes valen) o es null cuando el dato NO aparece; incorrecto si inventa, "
            "atribuye mal o pone null cuando el dato aparece claramente.\n"
            "- **numero_hablantes**: coincide con las marcas [HABLANTE_n] del texto (o null si no hay).\n"
            "- **citas_clave**: cada cita está en el texto (salvo puntuación) con el hablante que el texto indica.\n"
            "- Se juzga contra el texto que leyó el LLM: los errores de OCR/ASR ya se miden con CER/WER.")
    k = st.selectbox("Caso", range(len(casos)),
                     format_func=lambda j: f"{'✅' if casos.iloc[j]['id'] in listos else '⬜'} {casos.iloc[j]['id']} · {casos.iloc[j]['modalidad']}")
    if "guardado" in st.session_state:
        st.success(f"Caso {st.session_state.pop('guardado')} guardado")
    caso = casos.iloc[k]
    campos = json.loads(caso["campos_json"])
    # juicios ya guardados de este caso: se cargan en los controles para no pisarlos con los valores por defecto
    previos = {f["campo"]: f for _, f in hecho[hecho["id"] == caso["id"]].iterrows()}
    izq, der = st.columns(2)
    izq.markdown("**Texto fuente (lo que leyó el LLM)**")
    izq.text_area("fuente", caso["texto_fuente"], height=460, disabled=True, label_visibility="collapsed")
    der.markdown("**Extracción del LLM**")
    juicios = {}
    for campo, valor in campos.items():
        if campo == "modalidad":
            continue
        der.markdown(f"`{campo}` → {json.dumps(valor, ensure_ascii=False)}")
        a, b = der.columns([1, 2])
        previo = previos.get(campo)
        nota_previa = previo["nota"] if previo is not None and isinstance(previo["nota"], str) else ""
        juicios[campo] = (a.radio(campo, ["correcto", "incorrecto"], key=f"{caso['id']}_{campo}", horizontal=True,
                                  index=0 if previo is None or bool(previo["correcto"]) else 1,
                                  label_visibility="collapsed"),
                          b.text_input("nota", value=nota_previa, key=f"{caso['id']}_{campo}_n",
                                       label_visibility="collapsed", placeholder="nota si es incorrecto"))
    if previos:
        der.caption("Este caso ya estaba guardado: lo que ves son tus juicios anteriores.")
    if der.button("Guardar caso", type="primary"):
        nuevas = pd.DataFrame([{"id": caso["id"], "modalidad": caso["modalidad"], "campo": c,
                                "correcto": v == "correcto", "nota": n} for c, (v, n) in juicios.items()])
        hecho = pd.concat([hecho[hecho["id"] != caso["id"]], nuevas], ignore_index=True)
        hecho.to_csv(salida, index=False)
        st.session_state["guardado"] = caso["id"]
        st.rerun()
