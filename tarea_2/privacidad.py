"""Seudonimización de la Tarea 2: regex para identificadores, NER para candidatos a nombre y revisión humana.

Criterio (decidido por el autor): se seudonimizan las personas PRIVADAS y cualquier DNI, celular, correo o RUC de
persona natural. Se conservan las autoridades que firman en su cargo, los personajes históricos y los dirigentes
que hablan en una conferencia de prensa. El diccionario con los nombres reales (datos/raw/seudonimos.json) nunca
se versiona.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from urllib.parse import unquote

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


def diccionario(revision: pd.DataFrame, alias: pd.DataFrame | None = None) -> dict[str, str]:
    """Nombre real -> [PERSONA_n], solo para los candidatos marcados como 'privado'.

    `alias` (columnas `alias`, `candidato`) agrega otras grafías de la misma persona con su mismo seudónimo: por
    ejemplo, el nombre tal como figura en el título de un archivo de Commons cuando el NER lo leyó distinto.
    """
    privados = revision[revision["clase"] == "privado"]["candidato"].tolist()
    salida = {nombre: f"[PERSONA_{k}]" for k, nombre in enumerate(privados, 1)}
    if alias is not None:
        for otro, candidato in zip(alias["alias"], alias["candidato"]):
            if candidato in salida:
                salida[otro] = salida[candidato]
    return salida


def _plegar(s: str) -> str:
    """Minúsculas y sin tildes, para que 'Vásquez' y 'Vasquez' cuenten igual."""
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def coincidencias(texto: str, nombre: str, parecido: int = 85) -> list[tuple[int, int]]:
    """Tramos (inicio, fin) de `texto` donde aparece `nombre`, aunque el OCR o el ASR lo hayan escrito de otra forma.

    - Las palabras se separan también por guiones bajos (rutas de archivo) y se comparan sin tildes.
    - Una sola palabra de menos de 6 letras: solo coincidencia exacta.
    - Nombres de 2 o más palabras y 8 o más letras: ventanas de 1 a n+1 palabras comparadas sin espacios, para
      apellidos unidos o separados ("Quispeayala" / "Quispe Ayala"). Con 3 o más palabras, también los apellidos
      solos (el nombre sin la primera palabra).
    - Nombres de 6 o más letras pegados a otra palabra ("dequispeayala"): la palabra que los contiene. Se ignoran
      las palabras de más de 40 caracteres (imágenes en base64 de los notebooks).
    - Cada coincidencia se recorta al tramo de mejor parecido, para no llevarse palabras vecinas.
    """
    from rapidfuzz import fuzz
    fichas = list(re.finditer(r"[^\W_]+", texto))
    plegadas = [_plegar(f.group()) for f in fichas]
    partes = _plegar(nombre).split()
    n, unido = len(partes), "".join(partes)
    if n == 1 and len(unido) < 6:
        return [(f.start(), f.end()) for f, p in zip(fichas, plegadas) if p == unido]
    largo = n >= 2 and len(unido) >= 8
    sep = "" if largo or n == 1 else " "   # un nombre corto de varias palabras se compara con espacios
    objetivos = [sep.join(partes)] + (["".join(partes[1:])] if largo and n >= 3 else [])
    tamanos = range(1, n + 2) if largo else [n]
    puntaje = lambda a, b: max(fuzz.ratio(sep.join(plegadas[a:b]), o) for o in objetivos)
    tramos, i = [], 0
    while i < len(fichas):
        if len(unido) >= 6 and len(unido) < len(plegadas[i]) <= 40 and unido in plegadas[i]:
            tramos.append((fichas[i].start(), fichas[i].end()))
            i += 1
            continue
        tope = min(len(fichas), i + max(tamanos))
        if any(i + m <= len(fichas) and puntaje(i, i + m) >= parecido for m in tamanos):
            # el sub-tramo con mejor parecido dentro de la ventana; a igualdad, el más corto
            a, b = max(((a, b) for a in range(i, tope) for b in range(a + 1, tope + 1) if b - a in tamanos),
                       key=lambda ab: (puntaje(*ab), -(ab[1] - ab[0])))
            tramos.append((fichas[a].start(), fichas[b - 1].end()))
            i = b
        else:
            i += 1
    return tramos


def seudonimizar(texto: str, reemplazos: dict[str, str]) -> tuple[str, int]:
    """voz.seudonimizar (difuso, rapidfuzz >= 85) + las variantes que encuentra `coincidencias` + identificadores.

    Devuelve el texto y cuántos reemplazos hubo.
    """
    from voz import seudonimizar as _seud
    if not isinstance(texto, str) or not texto:
        return texto, 0
    salida, conteo = quitar_identificadores(texto)
    antes = len(re.findall(r"\[PERSONA_\d+\]", salida))
    # nombres compuestos primero, para que "Ana María Pérez" no quede a medias por "Ana María"
    for nombre in sorted(reemplazos, key=lambda n: -len(n.split())):
        salida = _seud(salida, {nombre: reemplazos[nombre]}, parecido=85)
        for a, b in reversed(coincidencias(salida, nombre)):
            salida = salida[:a] + reemplazos[nombre] + salida[b:]
    personas = len(re.findall(r"\[PERSONA_\d+\]", salida)) - antes
    return salida, personas + sum(conteo.values())


# Personajes históricos nombrados en los metadatos del corpus (descripciones, autores de Commons y descartes)
# que comparten nombre de pila con un privado
PUBLICOS_EN_METADATOS = ["Carlos J. Bachmann", "Bachmann, Carlos J.", "José Carlos Mariátegui"]


def fugas(textos: list[str], reemplazos: dict[str, str], publicos: list[str] = ()) -> list[str]:
    """Nombres privados que siguen apareciendo, con la misma comparación que `coincidencias`.

    - La codificación de URL se deshace y los nombres públicos de `publicos` se quitan antes de buscar.
    - Los fragmentos de menos de 4 letras ("Pad", "ula") no se buscan, porque no identifican a nadie y
      coinciden con palabras comunes. Igual se reemplazan al seudonimizar.
    """
    limpios = []
    for t in textos:
        t = unquote(t or "").replace("_", " ")
        for p in publicos:
            t = re.sub(re.escape(p), " ", t, flags=re.I)
        limpios.append(t)
    return [nombre for nombre in reemplazos
            if len(nombre.replace(" ", "")) >= 4 and any(coincidencias(t, nombre) for t in limpios)]
