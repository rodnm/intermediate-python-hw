"""Extracción estructurada con Gemini (response_schema) y chequeo automático de literalidad."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# 3.7-flash devolvió 503 (alta demanda) el 26/10; 3.6-flash se agregó como respaldo antes de ver resultados
MODELOS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]
SISTEMA = (
    "Extraes información de textos peruanos obtenidos por OCR o por reconocimiento de voz. Devuelve solo JSON "
    "según el esquema. Reglas: 1) Usa únicamente información presente en el texto; si un dato no aparece, usa null. "
    "2) No inventes ni completes nombres, fechas ni montos. 3) Copia emisor, destinatario, fecha, lugar y monto tal "
    "como aparecen. 4) El texto puede tener errores de OCR o ASR: no los corrijas en las citas. 5) Las marcas "
    "[PERSONA_n], [DNI], [TELEFONO] y [CORREO] son seudónimos: mantenlas tal cual. 6) tema: de 2 a 5 palabras; "
    "resumen: como máximo 50 palabras y solo con hechos del texto. 7) En audio, numero_hablantes solo si el texto "
    "trae marcas [HABLANTE_n]; citas_clave: hasta 3 frases textuales, con su marca de hablante si existe."
)


class Cita(BaseModel):
    hablante: str | None = Field(description="HABLANTE_n si el texto trae marcas de hablante; si no, null")
    cita: str = Field(description="frase textual del texto, sin corregir")


class ExtraccionDocumento(BaseModel):
    modalidad: Literal["documento"]
    tema: str
    resumen: str
    tipo_documento: str | None
    emisor: str | None
    destinatario: str | None
    fecha: str | None
    lugar: str | None
    monto: str | None


class ExtraccionAudio(BaseModel):
    modalidad: Literal["audio"]
    tema: str
    resumen: str
    numero_hablantes: int | None
    lugar: str | None
    citas_clave: list[Cita]


class ItemAudio(ExtraccionAudio):
    id: str


class LoteAudio(BaseModel):
    items: list[ItemAudio]


def _llamar(cliente, contenido: str, esquema, modelos=MODELOS):
    from google.genai import types
    ultimo = None
    for modelo in modelos:
        for intento in range(2):
            try:
                r = cliente.models.generate_content(
                    model=modelo, contents=contenido,
                    config=types.GenerateContentConfig(system_instruction=SISTEMA, temperature=0,
                                                       response_mime_type="application/json", response_schema=esquema))
                return esquema.model_validate_json(r.text), modelo
            except Exception as e:  # 429/503 o JSON inválido: esperar y reintentar; luego, modelo de respaldo
                ultimo = e
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    break  # modelo saturado: pasar enseguida al respaldo
                time.sleep(4 * (intento + 1))
    raise RuntimeError(f"El LLM no respondió: {ultimo}")


def extraer(cliente, id_: str, modalidad: str, texto: str, cache: Path) -> dict:
    """Una llamada por id (documentos y clips de entrevistas), con caché en disco."""
    destino = cache / f"{id_}.json"
    if destino.exists():
        return json.loads(destino.read_text(encoding="utf-8"))
    esquema = ExtraccionDocumento if modalidad == "documento" else ExtraccionAudio
    obj, modelo = _llamar(cliente, f"Modalidad: {modalidad}\nTexto:\n{texto}", esquema)
    fila = {"id": id_, "modalidad": modalidad, "campos_json": obj.model_dump_json(), "modelo_llm": modelo,
            "fecha": datetime.now().isoformat(timespec="seconds")}
    cache.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(fila, ensure_ascii=False), encoding="utf-8")
    return fila


def extraer_lote(cliente, items: list[tuple[str, str]], cache: Path) -> list[dict]:
    """Oraciones cortas de OpenSLR en lotes de 10: cada ítem lleva su id y se valida que vuelvan todos."""
    pendientes = [(i, t) for i, t in items if not (cache / f"{i}.json").exists()]
    if pendientes:
        texto = "\n".join(f"- id={i} | texto: {t}" for i, t in pendientes)
        obj, modelo = _llamar(cliente, "Modalidad: audio. Extrae un ítem por cada id:\n" + texto, LoteAudio)
        por_id = {it.id: it for it in obj.items}
        faltan = {i for i, _ in pendientes} - set(por_id)
        if faltan:
            raise RuntimeError(f"El lote no devolvió {faltan}")
        cache.mkdir(parents=True, exist_ok=True)
        for i, _ in pendientes:
            campos = por_id[i].model_dump(exclude={"id"})
            fila = {"id": i, "modalidad": "audio", "campos_json": json.dumps(campos, ensure_ascii=False),
                    "modelo_llm": modelo + " (lote de 10)", "fecha": datetime.now().isoformat(timespec="seconds")}
            (cache / f"{i}.json").write_text(json.dumps(fila, ensure_ascii=False), encoding="utf-8")
    return [json.loads((cache / f"{i}.json").read_text(encoding="utf-8")) for i, _ in items]


CAMPOS_LITERALES = ["emisor", "destinatario", "fecha", "lugar", "monto"]


def literalidad(campos: dict, texto: str) -> tuple[int, int]:
    """Cuántos valores factuales y citas aparecen (difuso, partial_ratio >= 90) en el texto fuente."""
    from rapidfuzz import fuzz
    valores = [campos.get(k) for k in CAMPOS_LITERALES if campos.get(k)]
    valores += [c["cita"] for c in campos.get("citas_clave", []) or [] if c.get("cita")]
    base = (texto or "").lower()
    ok = sum(fuzz.partial_ratio(str(v).lower(), base) >= 90 for v in valores)
    return ok, len(valores)
