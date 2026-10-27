"""OCR de la Tarea 2: PaddleOCR y Gemini como VLM con la misma interfaz, orden de lectura y limpieza para CER/WER."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

# Mismos modelos PP-OCRv6 por defecto, con el motor onnxruntime. En Windows con CPU, el backend oneDNN de
# Paddle 3.3 falla con PP-OCRv6 (NotImplementedError en ConvertPirAttribute2RuntimeAttribute), y sin oneDNN
# tarda 114 s por página; onnxruntime tarda 28 s en la misma página y devuelve las mismas cajas.
PADDLE_BASE = dict(lang="es", engine="onnxruntime")
PADDLE_SIN_GEO = dict(PADDLE_BASE, use_doc_orientation_classify=False, use_doc_unwarping=False,
                      use_textline_orientation=False)
PROMPT_VLM = ("Transcribe literalmente todo el texto visible de esta imagen, en orden de lectura. "
              "No corrijas ortografía, no resumas, no agregues nada. Si una palabra es ilegible, escribe [?]. "
              "Devuelve solo el texto.")


def crear_paddle(geometria: bool = True):
    from paddleocr import PaddleOCR
    return PaddleOCR(**(PADDLE_BASE if geometria else PADDLE_SIN_GEO))


def orden_lectura(cajas, textos) -> list[str]:
    """Une las cajas en renglones (solapamiento vertical > 50 %) y ordena de arriba abajo y de izquierda a derecha."""
    if len(textos) == 0:
        return []
    cajas = np.asarray(cajas, dtype=float)  # [x0, y0, x1, y1]
    orden = np.argsort(cajas[:, 1])
    renglones: list[list[int]] = []
    for i in orden:
        y0, y1 = cajas[i, 1], cajas[i, 3]
        for r in renglones:
            ry0, ry1 = np.median(cajas[r, 1]), np.median(cajas[r, 3])
            solape = min(y1, ry1) - max(y0, ry0)
            if solape > 0.5 * min(y1 - y0, ry1 - ry0):
                r.append(i)
                break
        else:
            renglones.append([i])
    renglones.sort(key=lambda r: np.median(cajas[r, 1]))
    return [" ".join(textos[i] for i in sorted(r, key=lambda j: cajas[j, 0])) for r in renglones]


def leer_paddle(ocr, ruta: Path, carpeta: Path | None = None) -> dict:
    """Corre PaddleOCR sobre una imagen. Si `carpeta` existe, guarda el JSON crudo y la imagen con cajas."""
    t0 = time.perf_counter()
    r = ocr.predict(str(ruta))[0]
    seg = time.perf_counter() - t0
    textos, puntajes, cajas = list(r["rec_texts"]), [float(x) for x in r["rec_scores"]], r["rec_boxes"]
    if carpeta is not None:
        carpeta.mkdir(parents=True, exist_ok=True)
        r.save_to_json(str(carpeta))
        r.save_to_img(str(carpeta))
    lineas = orden_lectura(cajas, textos)
    return {"texto": "\n".join(lineas), "n_cajas": len(textos),
            "puntaje_medio": float(np.mean(puntajes)) if puntajes else 0.0, "segundos": round(seg, 2)}


def recorte_eval(origen: Path, destino: Path, fraccion: float = 0.4) -> Path:
    """Región de evaluación fijada de antemano: el 40 % superior de la página (igual para todos los modelos)."""
    from PIL import Image
    destino.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(origen).convert("RGB")
    img.crop((0, 0, img.width, round(img.height * fraccion))).save(destino)
    return destino


def leer_gemini(cliente, ruta: Path, modelos: list[str]) -> dict:
    """Gemini como modelo de visión y lenguaje: transcripción literal de la imagen (con modelos de respaldo)."""
    from google.genai import types
    datos = ruta.read_bytes()
    ultimo = None
    for modelo in modelos:
        for intento in range(2):
            try:
                t0 = time.perf_counter()
                r = cliente.models.generate_content(
                    model=modelo, contents=[types.Part.from_bytes(data=datos, mime_type="image/png"), PROMPT_VLM],
                    config=types.GenerateContentConfig(temperature=0))
                return {"texto": (r.text or "").strip(), "segundos": round(time.perf_counter() - t0, 2), "modelo": modelo}
            except Exception as e:  # 503: pasar enseguida al respaldo; otros errores: esperar y reintentar
                ultimo = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    break
                time.sleep(5 * (intento + 1))
    raise RuntimeError(f"Gemini no respondió: {ultimo}")


# ───────────────────────────────────────────── limpieza común para CER/WER
MULETILLAS = {"eh", "em", "emm", "mm", "mmm", "ah", "ehh"}


def limpiar_para_metrica(texto: str, quitar_muletillas: bool = False) -> str:
    """Misma limpieza para referencia e hipótesis, antes de voz.normalizar.

    Quita marcas entre corchetes ([HABLANTE_1], [PERSONA_2], [?]), prefijos de hablante (H1:), sellos de
    tiempo y convierte dígitos a palabras (num2words) para no penalizar '20' contra 'veinte'.
    """
    from num2words import num2words
    from voz import normalizar
    t = re.sub(r"\[[^\]]*\]", " ", texto or "")
    t = re.sub(r"\bH\d+:", " ", t)
    t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="es") + " " if len(m.group()) < 10 else m.group(), t)
    t = normalizar(t)
    if quitar_muletillas:
        t = " ".join(p for p in t.split() if p not in MULETILLAS)
    return t


def cer(referencias: list[str], hipotesis: list[str]) -> float:
    import jiwer
    return jiwer.cer([limpiar_para_metrica(r) for r in referencias], [limpiar_para_metrica(h) or "∅" for h in hipotesis])


def wer(referencias: list[str], hipotesis: list[str]) -> float:
    import jiwer
    return jiwer.wer([limpiar_para_metrica(r, True) for r in referencias],
                     [limpiar_para_metrica(h, True) or "∅" for h in hipotesis])


def cer_lineas_ordenadas(referencia: str, hipotesis: str) -> float:
    """CER que no depende del orden de lectura: compara las líneas ordenadas alfabéticamente."""
    import jiwer
    ordenar = lambda t: " ".join(sorted(limpiar_para_metrica(l) for l in (t or "").splitlines() if l.strip()))
    return jiwer.cer(ordenar(referencia) or "∅", ordenar(hipotesis) or "∅")
