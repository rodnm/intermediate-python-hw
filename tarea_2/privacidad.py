"""Seudonimización de la Tarea 2: regex para identificadores, NER para candidatos a nombre y revisión humana.

Criterio (decidido por el autor): se seudonimizan las personas PRIVADAS y cualquier DNI, celular, correo o RUC de
persona natural. Se conservan las autoridades que firman en su cargo, los personajes históricos y los dirigentes
que hablan en una conferencia de prensa. El diccionario con los nombres reales (datos/raw/seudonimos.json) nunca
se versiona.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

PATRONES = {
    "dni": re.compile(r"(?i)(?<=dni)[\s:.Nº°n]*\d{8}\b"),
    "celular": re.compile(r"\b9\d{2}[\s-]?\d{3}[\s-]?\d{3}\b"),
    "correo": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "ruc_persona": re.compile(r"\b10\d{9}\b"),
}
MARCAS = {"dni": "[DNI]", "celular": "[TELEFONO]", "correo": "[CORREO]", "ruc_persona": "[RUC]"}


def quitar_identificadores(texto: str) -> tuple[str, dict]:
    """Reemplaza DNI, celulares, correos y RUC de persona natural. Devuelve el texto y el conteo por tipo."""
    conteo = {}
    for tipo, patron in PATRONES.items():
        texto, n = patron.subn(MARCAS[tipo] if tipo != "dni" else " " + MARCAS[tipo], texto)
        conteo[tipo] = n
    return texto, conteo


def _trozos(texto: str, max_palabras: int = 100):
    palabras = texto.split()
    for i in range(0, len(palabras), max_palabras):
        yield " ".join(palabras[i:i + max_palabras])


def candidatos_ner(textos: dict[str, str], ids_sin_contexto: set[str], umbral: float = 0.6) -> pd.DataFrame:
    """Personas (PER) que detecta mrm8488/bert-spanish-cased-finetuned-ner, agrupadas por nombre.

    En los ids de la muestra de evaluación no se muestra contexto, para no contaminar las referencias ciegas.
    """
    from rapidfuzz import fuzz
    from transformers import pipeline
    ner = pipeline("ner", model="mrm8488/bert-spanish-cased-finetuned-ner", aggregation_strategy="simple")
    hallados = defaultdict(lambda: {"ids": set(), "apariciones": 0, "contexto": ""})
    for id_, texto in textos.items():
        for trozo in _trozos(texto or ""):
            for e in ner(trozo):
                nombre = re.sub(r"\s+", " ", e["word"].replace("##", "")).strip(" .,;:")
                if e["entity_group"] != "PER" or e["score"] < umbral or len(nombre) < 3:
                    continue
                clave = next((k for k in hallados if fuzz.ratio(k.lower(), nombre.lower()) >= 88), nombre)
                h = hallados[clave]
                h["ids"].add(id_)
                h["apariciones"] += 1
                if not h["contexto"] and id_ not in ids_sin_contexto:
                    a, b = max(0, e["start"] - 50), e["end"] + 50
                    h["contexto"] = re.sub(r"\s+", " ", trozo[a:b])
    filas = [{"candidato": k, "apariciones": v["apariciones"], "ids": " ".join(sorted(v["ids"])),
              "contexto": v["contexto"]} for k, v in hallados.items()]
    return pd.DataFrame(filas, columns=["candidato", "apariciones", "ids", "contexto"]).sort_values(
        "apariciones", ascending=False, ignore_index=True)


def diccionario(revision: pd.DataFrame) -> dict[str, str]:
    """Nombre real -> [PERSONA_n], solo para los candidatos marcados como 'privado'."""
    privados = revision[revision["clase"] == "privado"]["candidato"].tolist()
    return {nombre: f"[PERSONA_{k}]" for k, nombre in enumerate(privados, 1)}


def seudonimizar(texto: str, reemplazos: dict[str, str]) -> tuple[str, int]:
    """voz.seudonimizar (difuso, rapidfuzz >= 85) + identificadores. Devuelve el texto y cuántos reemplazos hubo."""
    from voz import seudonimizar as _seud
    if not isinstance(texto, str) or not texto:
        return texto, 0
    salida, conteo = quitar_identificadores(texto)
    antes = len(re.findall(r"\[PERSONA_\d+\]", salida))
    # nombres compuestos primero, para que "Ana María Pérez" no quede a medias por "Ana María"
    for nombre in sorted(reemplazos, key=lambda n: -len(n.split())):
        salida = _seud(salida, {nombre: reemplazos[nombre]}, parecido=85)
    personas = len(re.findall(r"\[PERSONA_\d+\]", salida)) - antes
    return salida, personas + sum(conteo.values())


# Personajes históricos nombrados en los metadatos del corpus (descripciones, autores de Commons y descartes)
# que comparten nombre de pila con un privado
PUBLICOS_EN_METADATOS = ["Carlos J. Bachmann", "Bachmann, Carlos J.", "José Carlos Mariátegui"]


def fugas(textos: list[str], reemplazos: dict[str, str], publicos: list[str] = ()) -> list[str]:
    """Nombres privados que siguen apareciendo (comparación difusa por ventanas de palabras).

    - Los nombres públicos de `publicos` se quitan del texto antes de buscar.
    - Los fragmentos de menos de 4 letras ("Pad", "ula") no se buscan, porque no identifican a nadie y
      coinciden con palabras comunes. Igual se reemplazan al seudonimizar.
    - Una sola palabra de menos de 6 letras se busca exacta: con 4 o 5 letras, el umbral difuso de 85
      ya tolera un carácter de diferencia, y una palabra de 3 letras contenida en el nombre lo activaría.
    """
    from rapidfuzz import fuzz
    limpios = []
    for t in textos:
        for p in publicos:
            t = re.sub(re.escape(p), " ", t or "", flags=re.I)
        limpios.append(re.findall(r"\w+", t or ""))
    encontrados = []
    for nombre in reemplazos:
        n = len(nombre.split())
        if len(nombre) < 4:
            continue
        umbral = 100 if n == 1 and len(nombre) < 6 else 85
        for palabras in limpios:
            if any(fuzz.ratio(" ".join(palabras[i:i + n]).lower(), nombre.lower()) >= umbral
                   for i in range(len(palabras) - n + 1)):
                encontrados.append(nombre)
                break
    return encontrados
