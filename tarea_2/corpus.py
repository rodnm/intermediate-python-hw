"""Corpus de la Tarea 2: qué se descarga, de dónde, con qué licencia y cómo se prepara.

Una sola fuente de verdad: las tablas DOCUMENTOS y GRABACIONES generan fuentes.csv y atribucion.csv.
Las descripciones son propias (no copian nombres de personas privadas que traiga Commons).
Las descargas respetan robots.txt, usan un User-Agent descriptivo y esperan 1-2 s entre solicitudes.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import time
import urllib.parse
import urllib.request
import urllib.robotparser
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent
RAW = RAIZ / "datos" / "raw"
UA = {"User-Agent": "rodnm-tarea2/1.0 (https://github.com/rodnm/intermediate-python-hw; tarea academica)"}
COMMONS = "https://upload.wikimedia.org/wikipedia/commons/"
API = "https://commons.wikimedia.org/w/api.php"
OPENSLR73 = "https://www.openslr.org/resources/73/es_pe_{sexo}.zip"

# id, tipo, ruta en upload.wikimedia.org, época (solo manuscritos), descripción propia
DOCUMENTOS = [
    ("doc_001", "impreso", "6/6c/DS_N%C2%B0_001-81-AA_DECLARAN_SANTUARIO_HISTORICO_AREA_UBICADA_EN_EL_DISTRITO_DEMACHUPICCHU.pdf", None,
     "Decreto Supremo 001-81-AA que declara el Santuario Histórico de Machu Picchu (El Peruano, 1981), p. 1"),
    ("doc_002", "impreso", "6/64/Decreto_Supremo_046-2020-Presidencia_del_Consejo_de_Ministros_sobre_Toque_de_Queda_en_Per%C3%BA_por_Covid-19.pdf", None,
     "Decreto Supremo 046-2020-PCM sobre el toque de queda por COVID-19, p. 1"),
    ("doc_003", "impreso", "d/d0/DECRETO_DE_URGENCIA_No_002-2019_Presidencia_de_la_Rep%C3%BAblica_del_Per%C3%BA.pdf", None,
     "Decreto de Urgencia 002-2019 sobre las elecciones congresales, p. 1"),
    ("doc_004", "impreso", "4/48/Limpieza_y_desinfecci%C3%B3n_en_domicilios_Coronavirus.pdf", None,
     "Folleto del MINSA/DIGESA sobre limpieza y desinfección en domicilios (2020)"),
    ("doc_005", "impreso", "1/1f/Comunicado_oficial_N_21_sobre_el_fallecimiento_de_2_personas_por_covid-19_en_Per%C3%BA.jpg", None,
     "Comunicado oficial N.° 21 del MINSA (2020)"),
    ("doc_006", "impreso", "f/fd/Mensaje_a_la_Naci%C3%B3n_del_Presidente_del_Per%C3%BA_sobre_Estado_de_emergencia_en_Per%C3%BA_covid-19_15-03-20.pdf", None,
     "Mensaje a la Nación sobre el estado de emergencia (15-03-2020), p. 1"),
    ("doc_007", "impreso", "2/22/Liga_de_Futbol_de_Lima_Boletin_169.jpg", None,
     "Boletín 169 de la Liga de Fútbol de Lima (1935)"),
    ("doc_008", "impreso", "0/04/Acta_de_Creaci%C3%B3n_de_los_S%C3%ADmbolos_para_el_Distrito_de_Mache_-_Per%C3%B9.jpg", None,
     "Acta de creación de los símbolos del distrito de Mache (2023)"),
    ("doc_009", "foto", "6/69/Comunicado_horario_servicio_Lubarsa.jpg", None,
     "Foto de un comunicado impreso con el horario de una ruta de transporte (2026)"),
    ("doc_010", "foto", "curid:150148728", None,
     "Foto de titulares de periódicos en un puesto de venta (2024)"),
    ("doc_011", "foto", "6/63/Quiosco_en_Lima_%2822_de_Marzo%29.jpg", None,
     "Foto de portadas de diarios en un quiosco de Lima (2018)"),
    ("doc_012", "foto", "f/ff/Lima_%28Per%C3%BA%29_-_Restaurante_El_Embrujo_-_Men%C3%BA.jpg", None,
     "Foto del menú impreso de un restaurante de Lima (2012)"),
    ("doc_013", "foto", "e/e7/CARTEL_ALUSIVO_A_LAS_MEDIDAS_DE_PREVENCI%C3%93N_FRENTE_AL_COVID_19-_UNMSM.jpg", None,
     "Foto de un cartel de prevención frente al COVID-19 en la UNMSM (2022)"),
    ("doc_014", "foto", "6/63/N%C2%B01_Edition_Of_%22El_Peruano%22_Newspaper.jpg", None,
     "Foto en vitrina del número 1 del diario El Peruano (Museo Bolivariano de Pativilca, 2018)"),
    ("doc_015", "foto", "9/92/Huaca_Melgarejo_%28Fichero_Informativo%29.jpg", None,
     "Foto del panel informativo de la Huaca Melgarejo (2022)"),
    ("doc_016", "foto", "9/98/Afiche_informativo_sobre_la_vacunaci%C3%B3n_contra_la_COVID-19_en_Per%C3%BA.jpg", None,
     "Foto de un afiche informativo sobre la vacunación contra la COVID-19 (2021)"),
    ("doc_017", "manuscrito", "0/01/AR-ANH-EF-Secc.II-53-2.jpg", "XIX",
     "Carta manuscrita de Agustín Gamarra (1822)"),
    ("doc_018", "manuscrito", "c/ce/Carta_de_C%C3%A9sar_Vallejo_a_Jos%C3%A9_Mar%C3%ADa_Eguren.pdf", "XX",
     "Carta manuscrita de César Vallejo a José María Eguren"),
    ("doc_019", "manuscrito", "c/c4/4000004161-0003_-_Carta_de_Carlos_J._Bachmann._1912_junio_10%2C_Riberalta%3B_para_Ricardo_Palma%2C_Lima.pdf", "XX",
     "Carta manuscrita de Carlos J. Bachmann a Ricardo Palma (1912), p. 1"),
    ("doc_020", "manuscrito", "1/12/Miguel_Grau_-_Brev.jpg", "XIX",
     "Carta manuscrita de Miguel Grau (1879)"),
    ("doc_021", "manuscrito", "c/c2/AR-ANH-EF-Secc.II-69.jpg", "XIX",
     "Manuscrito de Nicolás de Piérola al prefecto de Lambayeque (1897)"),
    ("doc_022", "manuscrito", "3/35/AR-ANH-EF-Secc.II-67.jpg", "XIX",
     "Nota manuscrita de José Balta (1869)"),
    ("doc_023", "manuscrito", "9/95/AR-ANH-EF-Secc.II-55.jpg", "XIX",
     "Nota manuscrita del general Pedro Bermúdez (1850)"),
    ("doc_024", "manuscrito", "4/4b/AR-ANH-EF-Secc.II-64.jpg", "XIX",
     "Carta manuscrita de Miguel San Román a Agustín Gamarra (1840)"),
    ("doc_025", "manuscrito", "5/5f/AR-ANH-EF-Secc.II-63.jpg", "XIX",
     "Nota manuscrita de José Rufino Echenique (1843)"),
]

# Descartados en la revisión visual previa al sorteo (se reportan en descartes.csv)
DESCARTES_PREVIOS = [
    ("foto", "8/8b/Afiches_informativos_sobre_desastres_que_ocurrieron_en_Per%C3%BA.jpg",
     "casi sin texto legible: afiches fotografiados de lejos", "doc_014"),
    ("foto", "e/ee/Afiche_representativo.jpg", "casi sin texto legible: estandarte en un poste", "doc_015"),
    ("foto", "b/be/Afiche_Museo_de_Sitio_del_Parque_de_la_Muralla.jpg",
     "casi sin texto legible y en inglés", "doc_016"),
    ("manuscrito", "4/4c/Carta_de_C%C3%A9sar_Vallejo_a_Jos%C3%A9_Carlos_Mari%C3%A1tegui.pdf",
     "no es manuscrito: carta mecanografiada (solo la firma es a mano)", "doc_017"),
    ("manuscrito", "7/74/4000004161-0016_-_Carta_de_Carlos_J._Bachmann._1914_julio_4%2C_Maldonado%3B_para_Ricardo_Palma%2C_Lima.pdf",
     "no es manuscrito: carta mecanografiada", "doc_020"),
    ("foto", "1/10/Cartel_del_Ministerio_de_Cultura_sobre_la_Huaca_Palomino.jpg",
     "reserva descartada: cartel fotografiado de lejos, casi sin texto legible", "doc_014"),
    ("foto", "f/f7/Panel_informativo_Desaguadero%2C_Per%C3%BA_03.jpg",
     "reserva descartada: señal de ruta con muy poco texto", "doc_015"),
]

# Reservas por tipo, en orden de uso, por si un documento se descarta antes del sorteo
RESERVAS = {
    "impreso": [("b/bd/Lineamientos_traslado_y_cuarentena_de_personas_en_Estado_de_emergencia_por_Covid-19_13_abril.pdf", None,
                 "Lineamientos de traslado y cuarentena de la PCM (2020), p. 1")],
    "foto": [("3/34/Panel_informativo_de_Waka_Sapantiana%2C_2026.jpg", None,
              "Foto del panel informativo de la Waka Sapantiana (2026)")],
    "manuscrito": [],
}

# id, ruta en upload.wikimedia.org, descripción propia, hablantes declarados por la fuente (None si no lo dice)
GRABACIONES = [
    ("rec_01", "0/0d/Entrevista_1_en_la_Av._Rivag%C3%BCero_-_El_Agustino.ogg",
     "Entrevista en la calle a una vendedora en El Agustino, Lima (2022)", 2),
    ("rec_02", "d/df/Entrevista_2_en_la_Av._Rivag%C3%BCero_-_El_Agustino.ogg",
     "Entrevista en la calle a dos vendedoras en El Agustino, Lima (2022)", 3),
    ("rec_03", "b/be/Entrevista_1%2C.wav", "Entrevista grabada por un estudiante de la UNMSM (1)", None),
    ("rec_04", "5/52/Entrevista_2.wav", "Entrevista grabada por un estudiante de la UNMSM (2)", None),
    ("rec_05", "1/10/Entrevista_3.wav", "Entrevista grabada por un estudiante de la UNMSM (3)", None),
    ("rec_06", "8/86/Entrevista_4.wav", "Entrevista grabada por un estudiante de la UNMSM (4)", None),
    ("rec_07", "curid:121219890",
     "Dirigente de una organización de mujeres indígenas en conferencia de prensa (2022)", None),
    ("rec_08", "curid:149472732",
     "Testimonio de una mujer de Junín", None),
]

_robots: dict[str, urllib.robotparser.RobotFileParser] = {}
_ultima = [0.0]


def _permitido(url: str) -> bool:
    """Consulta robots.txt del host (una vez por host)."""
    p = urllib.parse.urlsplit(url)
    host = f"{p.scheme}://{p.netloc}"
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser(host + "/robots.txt")
        try:
            with urllib.request.urlopen(urllib.request.Request(host + "/robots.txt", headers=UA), timeout=30) as r:
                rp.parse(r.read().decode("utf-8", "ignore").splitlines())
        except Exception:
            rp.parse([])  # sin robots.txt accesible: se asume permitido
        _robots[host] = rp
    return _robots[host].can_fetch(UA["User-Agent"], url)


def obtener(url: str, rango: tuple[int, int] | None = None, metodo: str = "GET", api: bool = False) -> bytes:
    """GET respetuoso: robots.txt, User-Agent descriptivo, pausa de 1-2 s y hasta 3 intentos.

    `api=True` es solo para la API de MediaWiki: su robots.txt excluye /w/ para rastreadores web, y la API
    se rige por su propia política de uso (https://www.mediawiki.org/wiki/API:Etiquette): User-Agent
    identificable, solicitudes en serie y pausas. Eso es lo que se hace aquí.
    """
    if not api and not _permitido(url):
        raise PermissionError(f"robots.txt no permite {url}")
    espera = random.uniform(1, 2) - (time.time() - _ultima[0])
    if espera > 0:
        time.sleep(espera)
    cab = dict(UA) | ({"Range": f"bytes={rango[0]}-{rango[1]}"} if rango else {})
    for intento in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=cab, method=metodo), timeout=120) as r:
                _ultima[0] = time.time()
                return r.read() if metodo == "GET" else json.dumps(dict(r.headers)).encode()
        except Exception:
            if intento == 2:
                raise
            time.sleep(3 * (intento + 1))


def ruta_real(ruta: str) -> str:
    """Resuelve las entradas "curid:<pageid>" a su ruta en upload.wikimedia.org.

    Se usan para los archivos cuyo nombre en Commons lleva el nombre de una persona privada: así ese nombre no
    queda en el código versionado. La respuesta de la API se guarda en datos/raw/, que no se versiona.
    """
    if not ruta.startswith("curid:"):
        return ruta
    cache = RAW / "rutas_curid.json"
    rutas = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    pageid = ruta.split(":", 1)[1]
    if pageid not in rutas:
        q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url",
                                    "pageids": pageid})
        url = json.loads(obtener(f"{API}?{q}", api=True))["query"]["pages"][pageid]["imageinfo"][0]["url"]
        rutas[pageid] = url.split("?", 1)[0].removeprefix(COMMONS)   # la API agrega parámetros utm_* a la URL
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rutas, ensure_ascii=False), encoding="utf-8")
    return rutas[pageid]


def titulo_commons(ruta: str) -> str:
    return urllib.parse.unquote(ruta_real(ruta).rsplit("/", 1)[1]).replace("_", " ")


def metadatos_commons(rutas: list[str]) -> pd.DataFrame:
    """Licencia, autor, fecha y pageid de cada archivo, leídos de la API de Commons (de a 40 títulos)."""
    filas = []
    titulos = [titulo_commons(r) for r in rutas]
    for i in range(0, len(titulos), 40):
        lote = titulos[i:i + 40]
        q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo",
                                    "iiprop": "extmetadata|size|mime", "titles": "|".join("File:" + t for t in lote)})
        datos = json.loads(obtener(f"{API}?{q}", api=True))
        normal = {n["to"]: n["from"] for n in datos["query"].get("normalized", [])}
        for pagina in datos["query"]["pages"].values():
            titulo = pagina["title"]
            ii = pagina["imageinfo"][0]
            em = ii.get("extmetadata", {})
            valor = lambda k: (em.get(k, {}).get("value") or "").strip()
            filas.append({"titulo": normal.get(titulo, titulo).removeprefix("File:"), "pageid": pagina["pageid"],
                          "licencia": valor("LicenseShortName"), "autor": _sin_html(valor("Artist")),
                          "fecha_original": _sin_html(valor("DateTimeOriginal"))[:40],
                          "ancho": ii.get("width"), "alto": ii.get("height"), "mime": ii.get("mime"),
                          "duracion_seg": ii.get("duration")})
    return pd.DataFrame(filas)


def _sin_html(texto: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", texto)).strip()


def sha256(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def descargar(ruta_commons: str, destino: Path) -> Path:
    """Descarga un archivo de Commons si todavía no está en disco."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        destino.write_bytes(obtener(COMMONS + ruta_real(ruta_commons)))
    return destino


def rasterizar(origen: Path, destino: Path, dpi: int = 200, pagina: int = 0) -> Path:
    """PDF -> PNG de la página indicada (pypdfium2, como en la sesión 10). Las imágenes se copian tal cual."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        return destino
    if origen.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(origen))
        pdf[pagina].render(scale=dpi / 72).to_pil().convert("RGB").save(destino)
    else:
        from PIL import Image
        Image.open(origen).convert("RGB").save(destino)
    return destino


# ───────────────────────────────────────────── OpenSLR 73 por HTTP Range (patrón de datos.py, sesión 13)
class ZipRemoto(io.RawIOBase):
    """Archivo remoto que se lee por partes: zipfile solo pide los bytes que necesita."""

    BLOQUE = 1 << 20

    def __init__(self, url):
        self.url, self.pos, self.buf_ini, self.buf = url, 0, 0, b""
        self.size = int(json.loads(obtener(url, metodo="HEAD"))["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        fin = self.pos + n
        if not (self.buf_ini <= self.pos and fin <= self.buf_ini + len(self.buf)):
            self.buf_ini = self.pos
            self.buf = obtener(self.url, (self.pos, min(self.pos + max(n, self.BLOQUE), self.size) - 1))
        datos = self.buf[self.pos - self.buf_ini:fin - self.buf_ini]
        self.pos += len(datos)
        return datos


def bajar_openslr73(por_hablante: int = 4, hablantes_por_sexo: int = 15, semilla: int = 7) -> pd.DataFrame:
    """Oraciones de español peruano leído (OpenSLR 73, CC BY-SA 4.0), elegidas con semilla fija."""
    import soundfile as sf
    from scipy.signal import resample_poly

    destino = RAW / "audio" / "openslr73"
    destino.mkdir(parents=True, exist_ok=True)
    indice = destino / "seleccion.csv"
    if indice.exists():
        return pd.read_csv(indice)
    filas = []
    for sexo in ["female", "male"]:
        z = zipfile.ZipFile(ZipRemoto(OPENSLR73.format(sexo=sexo)))
        lineas = z.read("line_index.tsv").decode("utf-8").strip().splitlines()
        tabla = pd.DataFrame([l.split("\t") for l in lineas], columns=["archivo", "texto"])
        tabla["hablante"] = tabla["archivo"].str.rsplit("_", n=1).str[0]
        tabla = tabla[tabla["texto"].str.split().str.len().between(6, 20)]
        rng = random.Random(semilla)
        hablantes = sorted(tabla["hablante"].unique())
        rng.shuffle(hablantes)
        for h in hablantes[:hablantes_por_sexo]:
            for _, fila in tabla[tabla["hablante"] == h].sample(por_hablante, random_state=semilla).iterrows():
                x, sr = sf.read(io.BytesIO(z.read(f"{fila['archivo']}.wav")), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != 16_000:
                    g = math.gcd(sr, 16_000)
                    x = resample_poly(x, 16_000 // g, sr // g).astype("float32")
                sf.write(destino / f"{fila['archivo']}.wav", x, 16_000)
                filas.append({"archivo": fila["archivo"], "texto_corpus": fila["texto"].strip(), "hablante": h,
                              "sexo": "mujer" if sexo == "female" else "hombre",
                              "zip": f"es_pe_{sexo}.zip"})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(indice, index=False)
    return tabla


def cortar_en_clips(audio: np.ndarray, max_seg: float = 60.0, sr: int = 16_000) -> list[tuple[float, float]]:
    """Corta una grabación larga en clips de <= max_seg que cubren TODA la grabación.

    Solo usa el VAD (no el ASR): los cortes caen en la mitad de una pausa entre tramos de voz.
    """
    from voz import tramos_de_voz

    total = len(audio) / sr
    tramos = tramos_de_voz(audio, silencio_min=0.5, max_tramo=25.0)
    if not tramos:
        return [(0.0, round(total, 2))]
    cortes, inicio = [], 0.0
    for (a0, b0), (a1, b1) in zip(tramos, tramos[1:]):
        medio = (b0 + a1) / 2
        if b1 - inicio > max_seg:  # el siguiente tramo ya no cabe en el clip actual
            cortes.append((inicio, medio))
            inicio = medio
    cortes.append((inicio, total))
    salida = []
    for a, b in cortes:  # un clip que aun así pase de max_seg (un tramo larguísimo) se parte en partes iguales
        n = max(1, math.ceil((b - a) / max_seg))
        paso = (b - a) / n
        salida += [(round(a + i * paso, 2), round(a + (i + 1) * paso, 2)) for i in range(n)]
    return salida


def textos_de_clips(clips: pd.DataFrame, segmentos: pd.DataFrame) -> pd.DataFrame:
    """Reparte los segmentos diarizados de cada grabación entre sus clips (por el punto medio del segmento).

    Devuelve por clip: el texto con marcas [HABLANTE_n] (turnos consecutivos unidos) y cuántos hablantes hay.
    La numeración de hablantes es la de la grabación completa: H1 es quien habla primero.
    """
    filas = []
    for f in clips.itertuples():
        s = segmentos[(segmentos.grabacion == f.grabacion)
                      & ((segmentos.inicio + segmentos.fin) / 2 >= f.inicio_seg)
                      & ((segmentos.inicio + segmentos.fin) / 2 < f.fin_seg)].sort_values("inicio")
        turnos = []
        for h, t in zip(s.hablante, s.texto):
            marca = f"[HABLANTE_{str(h).lstrip('H')}]"
            if turnos and turnos[-1][0] == marca:
                turnos[-1][1].append(t)
            else:
                turnos.append([marca, [t]])
        filas.append({"id": f.id, "texto_asr": " ".join(f"{m} {' '.join(x)}" for m, x in turnos),
                      "n_hablantes": s.hablante.nunique()})
    return pd.DataFrame(filas)
