"""Paleta Catppuccin Mocha para el dashboard y los gráficos de los notebooks.

Colores de https://catppuccin.com/palette/ (sabor Mocha). Se eligieron y validaron para que cada par se distinga
también con daltonismo. Al importar el módulo se registra la plantilla de Plotly "catppuccin".
"""
import plotly.graph_objects as go
import plotly.io as pio

# Superficies y texto
FONDO = "#1e1e2e"        # base
FONDO_ALTO = "#313244"   # surface0
BORDE = "#45475a"        # surface1
TEXTO = "#cdd6f4"        # text
GRIS = "#7f849c"         # overlay1

# Un color por medio (ΔE 17.2 entre ambos con deuteranopía)
COLORES_MEDIO = {"El Comercio": "#f9e2af", "La Republica": "#f38ba8"}  # yellow, red

# Series que no representan a un medio
SERIE = "#89b4fa"        # blue
CONTRASTE = "#fab387"    # peach: se distingue de blue con daltonismo (ΔE 17.6)
SECUENCIAL = ["#7f849c", "#89b4fa", "#cdd6f4"]  # overlay1 → blue → text (un solo tono)

_EJES = dict(gridcolor=BORDE, linecolor=BORDE, zerolinecolor=BORDE)
pio.templates["catppuccin"] = go.layout.Template(layout=dict(
    paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TEXTO), colorway=[SERIE, CONTRASTE],
    colorscale=dict(sequential=SECUENCIAL), xaxis=_EJES, yaxis=_EJES, legend=dict(bgcolor=FONDO),
    hoverlabel=dict(bgcolor=FONDO_ALTO, font=dict(color=TEXTO))))


def mapa_secuencial():
    """Colormap de matplotlib con la escala secuencial, para los heatmaps de seaborn."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("catppuccin", SECUENCIAL)


def estilo_matplotlib():
    """Fondo, ejes y texto de Catppuccin para las figuras de matplotlib y seaborn."""
    import matplotlib as mpl

    mpl.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                         "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.edgecolor": BORDE,
                         "xtick.color": TEXTO, "ytick.color": TEXTO})
