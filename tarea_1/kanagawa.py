"""Paleta Kanagawa (tema Wave) para el dashboard y los gráficos de los notebooks.

Colores de https://github.com/rebelot/kanagawa.nvim (lua/kanagawa/colors.lua). Se eligieron y validaron para que
cada par se distinga también con daltonismo. Al importar el módulo se registra la plantilla de Plotly "kanagawa".
"""
import plotly.graph_objects as go
import plotly.io as pio

# Superficies y texto, con los mismos roles que el tema Wave
FONDO = "#1F1F28"        # sumiInk3
FONDO_ALTO = "#2A2A37"   # sumiInk4
BORDE = "#363646"        # sumiInk5
TEXTO = "#DCD7BA"        # fujiWhite
GRIS = "#727169"         # fujiGray

# Un color por medio: el par "autumn" de Kanagawa (ΔE 19.7 entre ambos con deuteranopía)
COLORES_MEDIO = {"El Comercio": "#DCA561", "La Republica": "#C34043"}  # autumnYellow, autumnRed

# Series que no representan a un medio
SERIE = "#7E9CD8"        # crystalBlue
CONTRASTE = "#FFA066"    # surimiOrange: se distingue de crystalBlue con daltonismo (ΔE 18.8)
SECUENCIAL = ["#658594", "#7FB4CA", "#A3D4D5"]  # dragonBlue → springBlue → lightBlue (un solo tono)

_EJES = dict(gridcolor=BORDE, linecolor=BORDE, zerolinecolor=BORDE)
pio.templates["kanagawa"] = go.layout.Template(layout=dict(
    paper_bgcolor=FONDO, plot_bgcolor=FONDO, font=dict(color=TEXTO), colorway=[SERIE, CONTRASTE],
    colorscale=dict(sequential=SECUENCIAL), xaxis=_EJES, yaxis=_EJES, legend=dict(bgcolor=FONDO),
    hoverlabel=dict(bgcolor=FONDO_ALTO, font=dict(color=TEXTO))))


def mapa_secuencial():
    """Colormap de matplotlib con la escala secuencial, para los heatmaps de seaborn."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("kanagawa", SECUENCIAL)


def estilo_matplotlib():
    """Fondo, ejes y texto de Kanagawa para las figuras de matplotlib y seaborn."""
    import matplotlib as mpl

    mpl.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                         "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.edgecolor": BORDE,
                         "xtick.color": TEXTO, "ytick.color": TEXTO})
