"""Paleta del proyecto para el dashboard y los gráficos de los notebooks.

Rampa de 8 colores, del azul petróleo (#003d5c) al ámbar (#ffa600), con un negro (#242424) y un blanco (#fcfcfc)
como neutros. El fondo es el negro porque el amarillo de El Comercio solo alcanza buen contraste sobre fondo oscuro
(7.9:1, contra 1.9:1 sobre el blanco). Cada par se validó para que se distinga también con daltonismo. Al importar
el módulo se registra la plantilla de Plotly "paleta".
"""
import plotly.graph_objects as go
import plotly.io as pio

RAMPA = ["#003d5c", "#31497e", "#674f95", "#a14e9a", "#d44c8d", "#f9596f", "#ff7a47", "#ffa600"]

# Superficies y texto: los neutros del proyecto y los dos azules más oscuros de la rampa
FONDO = "#242424"        # negro
FONDO_ALTO = "#003d5c"
BORDE = "#31497e"
TEXTO = "#fcfcfc"        # blanco (15:1 sobre el fondo)
GRIS = "#674f95"

# Colores por categoría de la Tarea 2 (tonos de la rampa con buen contraste sobre el fondo negro)
COLORES_TIPO = {"impreso": "#ffa600", "foto": "#f9596f", "manuscrito": "#a14e9a"}
COLORES_FUENTE = {"Wikimedia Commons": "#d44c8d", "OpenSLR 73": "#ffa600"}
COLORES_MODELO = {"paddle_base": "#ffa600", "paddle_sin_geo": "#ff7a47", "gemini_vlm": "#a14e9a",
                  "fw_small": "#ffa600", "fw_turbo": "#f9596f", "parakeet": "#a14e9a"}

# Series sueltas (ΔE 17.2 entre ambas con deuteranopía; 3.9:1 y 6.0:1 contra el fondo)
SERIE = "#d44c8d"
CONTRASTE = "#ff7a47"
SECUENCIAL = RAMPA[1:]  # de oscuro a claro, en el orden de la rampa

_EJES = dict(gridcolor=BORDE, linecolor=BORDE, zerolinecolor=BORDE)
pio.templates["paleta"] = go.layout.Template(layout=dict(
    paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TEXTO), colorway=[SERIE, CONTRASTE],
    colorscale=dict(sequential=SECUENCIAL), xaxis=_EJES, yaxis=_EJES, legend=dict(bgcolor=FONDO),
    hoverlabel=dict(bgcolor=FONDO_ALTO, font=dict(color=TEXTO))))


def mapa_secuencial():
    """Colormap de matplotlib con la escala secuencial, para los heatmaps de seaborn."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("paleta", SECUENCIAL)


def estilo_matplotlib():
    """Fondo, ejes y texto de la paleta para las figuras de matplotlib y seaborn."""
    import matplotlib as mpl

    mpl.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                         "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.edgecolor": BORDE,
                         "xtick.color": TEXTO, "ytick.color": TEXTO})
