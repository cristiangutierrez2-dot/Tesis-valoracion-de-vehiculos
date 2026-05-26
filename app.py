import pickle
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import shap
import lime.lime_tabular
import re

# ── Configuracion ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "AutoTasar · Tesis UDP",
    page_icon   = "🚘",
    layout      = "wide",
    initial_sidebar_state = "collapsed"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #060912;
    color: #E8EAF0;
  }

  /* Ocultar menu hamburguesa */
  #MainMenu, footer, header { visibility: hidden; }

  /* Hero */
  .hero {
    background: linear-gradient(135deg, #0A1628 0%, #0F2040 50%, #0A1628 100%);
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 60px 40px;
    text-align: center;
    margin-bottom: 48px;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 50% 50%, #1F3864 0%, transparent 60%);
    opacity: 0.3;
  }
  .hero-tag {
    display: inline-block;
    background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.3);
    color: #60A5FA;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 99px;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 20px;
  }
  .hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: white;
    line-height: 1.2;
    margin: 0 0 16px;
  }
  .hero-subtitle {
    font-size: 1rem;
    color: #8090B0;
    max-width: 600px;
    margin: 0 auto 24px;
    line-height: 1.7;
  }
  .hero-authors {
    font-size: 0.82rem;
    color: #4B5580;
  }
  .hero-metrics {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 32px;
    flex-wrap: wrap;
  }
  .hero-metric {
    text-align: center;
  }
  .hero-metric-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #3B82F6;
  }
  .hero-metric-lbl {
    font-size: 0.75rem;
    color: #6B7590;
    margin-top: 2px;
  }

  /* Seccion */
  .seccion {
    margin: 48px 0;
  }
  .seccion-tag {
    font-size: 0.72rem;
    font-weight: 600;
    color: #3B82F6;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .seccion-titulo {
    font-size: 1.6rem;
    font-weight: 700;
    color: white;
    margin: 0 0 12px;
  }
  .seccion-desc {
    font-size: 0.9rem;
    color: #8090B0;
    line-height: 1.7;
    max-width: 700px;
    margin-bottom: 28px;
  }

  /* Cards pipeline */
  .pipeline {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin: 24px 0;
  }
  .pipeline-card {
    background: #0F1828;
    border: 1px solid #1E2E48;
    border-radius: 12px;
    padding: 16px;
  }
  .pipeline-num {
    font-size: 1.4rem;
    font-weight: 800;
    color: #2563EB;
    margin-bottom: 6px;
  }
  .pipeline-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #C8CCE0;
    margin-bottom: 4px;
  }
  .pipeline-desc {
    font-size: 0.78rem;
    color: #6B7590;
    line-height: 1.5;
  }

  /* Metricas modelo */
  .modelo-cards {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin: 24px 0;
  }
  .modelo-card {
    background: #0F1828;
    border: 1px solid #1E2E48;
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
  }
  .modelo-card-label {
    font-size: 0.75rem;
    color: #6B7590;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
  }
  .modelo-card-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #3B82F6;
  }
  .modelo-card-sub {
    font-size: 0.78rem;
    color: #4B5580;
    margin-top: 4px;
  }

  /* Simulador */
  .sim-box {
    background: #0A0F1E;
    border: 1px solid #1E2E48;
    border-radius: 20px;
    padding: 32px;
    margin: 24px 0;
  }
  .price-card {
    background: linear-gradient(135deg, #0F1A30, #0A1428);
    border: 1px solid #1E3A5F;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 20px 0;
  }
  .price-label {
    color: #6B8AB0;
    font-size: 0.75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0 0 8px;
  }
  .price-value {
    color: #3B82F6;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 6px;
  }
  .price-sub {
    color: #4B5580;
    font-size: 0.82rem;
    margin: 0;
  }

  /* Boton */
  .stButton > button {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 12px 32px !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563EB, #3B82F6) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(37,99,235,0.3) !important;
  }

  /* Divider */
  .divider {
    border: none;
    border-top: 1px solid #1E2535;
    margin: 48px 0;
  }

  /* Footer */
  .footer {
    text-align: center;
    padding: 32px 0 20px;
    border-top: 1px solid #1E2535;
    color: #3B4560;
    font-size: 0.8rem;
    line-height: 1.8;
  }

  /* Selectbox y inputs dark */
  .stSelectbox > div > div {
    background: #0F1828 !important;
    border-color: #1E2E48 !important;
    color: #E8EAF0 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo ─────────────────────────────────────────────
@st.cache_resource
def cargar_todo():
    with open("modelo_xgb.pkl", "rb") as f:
        bundle = pickle.load(f)

    xgb   = bundle["xgb"]
    # Convertir a dict puro por si viene como pd.Series
    ENC = {c: dict(v) if hasattr(v, "items") else v 
           for c, v in bundle["ENCODING_MAPS"].items()}
    GM = float(bundle["GLOBAL_MEAN"])
    FEATS = bundle["FEATURES_CAT"]
    ALL   = ["año", "kilometraje"] + FEATS

    # Muestra sintetica para LIME
    rng  = np.random.default_rng(42)
    rows = []
    for _ in range(200):
        vec = [float(rng.integers(2005, 2024)),
               float(rng.integers(10000, 200000))]
        for c in FEATS:
            vals = list(ENC[c].values())
            vec.append(float(rng.choice(vals)))
        rows.append(vec)
    X_synt = np.array(rows)

    exp_shap = shap.TreeExplainer(xgb)
    exp_lime = lime.lime_tabular.LimeTabularExplainer(
        training_data         = X_synt,
        feature_names         = ALL,
        mode                  = "regression",
        discretize_continuous = False,
        random_state          = 42,
    )
    return bundle, xgb, ENC, GM, FEATS, ALL, exp_shap, exp_lime

bundle, xgb, ENC, GM, FEATS, ALL_FEATURES, exp_shap, exp_lime = cargar_todo()

GAMA          = bundle["gama_map"]
MARCAS        = bundle["TODAS_MARCAS"]
CAT_MODELOS   = bundle["catalogo_modelos"]
CAT_PAIS      = bundle["catalogo_pais_por_marca"]
CAT_REGION    = bundle["catalogo_region_por_pais"]
TRANSMISIONES = bundle["TODAS_TRANSMISIONES"]
TRACCIONES    = bundle["TODAS_TRACCIONES"]
CARROCERIAS   = bundle["TODAS_CARROCERIAS"]
COMBUSTIBLES  = bundle["TODOS_COMBUSTIBLES"]
COLORES       = bundle["TODOS_COLORES"]

# ── Helpers graficos ──────────────────────────────────────────
def shap_a_clp(sv, base):
    b = np.expm1(base)
    return np.array([np.expm1(base + s) - b for s in sv])

def lime_a_clp(pw, pred):
    b = np.expm1(pred)
    return np.array([np.expm1(pred + p) - b for p in pw])

def fmt_clp(v):
    m = v / 1_000_000
    return f'{"+" if m >= 0 else ""}{m:.1f}M'

DARK = "#060912"
AX   = "#0F1828"
GRN  = "#22C55E"
RED  = "#EF4444"
TXT  = "#C8CCE0"
GRID = "#1E2535"

def grafico(efectos, etiquetas, titulo):
    n   = len(etiquetas)
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(AX)
    colores = [GRN if v > 0 else RED for v in efectos]
    ax.barh(etiquetas[::-1], efectos[::-1],
            color=colores[::-1], edgecolor=DARK,
            linewidth=0.4, height=0.65)
    rng = max(abs(efectos)) if len(efectos) else 1
    off = rng * 0.012
    for i, val in enumerate(efectos[::-1]):
        ax.text(val + (off if val >= 0 else -off), i,
                fmt_clp(val), va="center",
                ha="left" if val >= 0 else "right",
                fontsize=9, color=TXT, fontweight="500")
    ax.axvline(0, color="#2D3A55", linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_xlabel("Impacto en el precio (CLP)", fontsize=10, color=TXT)
    ax.set_title(titulo, fontsize=11, color="white", pad=12, fontweight="bold")
    ax.tick_params(colors=TXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.4, alpha=0.5)
    ax.legend(handles=[
        mpatches.Patch(color=GRN, label="Sube el precio"),
        mpatches.Patch(color=RED, label="Baja el precio"),
    ], fontsize=9, loc="lower right",
       facecolor=AX, edgecolor=GRID, labelcolor=TXT)
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# SECCION 1 — HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-tag">Magister en Negocios Digitales · UDP · 2025</div>
  <h1 class="hero-title">
    Interpretabilidad de modelos predictivos<br>
    aplicados a la valoracion de vehiculos usados en Chile
  </h1>
  <p class="hero-subtitle">
    Un analisis basado en tecnicas de explicabilidad local y global
    mediante LIME y SHAP aplicadas sobre modelos Random Forest y XGBoost,
    entrenados con 117.677 registros de ChileAutos (2024).
  </p>
  <p class="hero-authors">
    Echeverria P. &nbsp;·&nbsp; Gonzalez F. &nbsp;·&nbsp;
    Gutierrez C. &nbsp;·&nbsp; Valenzuela P.<br>
    <span style="color:#2D3A55">Profesor guia: Francisco Alessandri</span>
  </p>
  <div class="hero-metrics">
    <div class="hero-metric">
      <div class="hero-metric-val">117.677</div>
      <div class="hero-metric-lbl">Registros analizados</div>
    </div>
    <div class="hero-metric">
      <div class="hero-metric-val">92,43%</div>
      <div class="hero-metric-lbl">R² XGBoost</div>
    </div>
    <div class="hero-metric">
      <div class="hero-metric-val">2</div>
      <div class="hero-metric-lbl">Tecnicas XAI</div>
    </div>
    <div class="hero-metric">
      <div class="hero-metric-val">12</div>
      <div class="hero-metric-lbl">Variables del modelo</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECCION 2 — EL PROBLEMA
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="seccion">
  <div class="seccion-tag">El problema</div>
  <h2 class="seccion-titulo">¿Por que es difícil valorar un auto usado?</h2>
  <p class="seccion-desc">
    El mercado automotriz chileno registra mas de 1.000.000 de transferencias
    anuales de vehiculos usados, generando una necesidad creciente de mecanismos
    de valoracion precisos y transparentes. El precio de un vehiculo usado depende
    de decenas de variables que interactuan de forma no lineal: el modelo, el año,
    el kilometraje, la traccion, el combustible y muchas mas. Los metodos
    estadisticos tradicionales no logran capturar estas relaciones complejas,
    mientras que los modelos de aprendizaje automatico, aunque mas precisos,
    son cajas negras que no explican sus predicciones.
  </p>
  <p class="seccion-desc">
    Esta investigacion resuelve ambos problemas: construye modelos de alta
    precision y los hace completamente transparentes mediante tecnicas de
    Inteligencia Artificial Explicable (XAI).
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECCION 3 — METODOLOGIA
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="seccion">
  <div class="seccion-tag">Metodologia</div>
  <h2 class="seccion-titulo">¿Como funciona el modelo?</h2>
  <p class="seccion-desc">
    El pipeline combina un preprocesamiento riguroso con dos modelos de ensamble
    y dos tecnicas de explicabilidad complementarias.
  </p>
  <div class="pipeline">
    <div class="pipeline-card">
      <div class="pipeline-num">01</div>
      <div class="pipeline-title">Limpieza de datos</div>
      <div class="pipeline-desc">Estandarizacion, imputacion y correccion de 117.677 registros de ChileAutos 2024.</div>
    </div>
    <div class="pipeline-card">
      <div class="pipeline-num">02</div>
      <div class="pipeline-title">Target Encoding</div>
      <div class="pipeline-desc">Variables categoricas convertidas a valores numericos preservando informacion de precio promedio.</div>
    </div>
    <div class="pipeline-card">
      <div class="pipeline-num">03</div>
      <div class="pipeline-title">Transformacion logaritmica</div>
      <div class="pipeline-desc">El precio se transforma con ln(1+precio) para corregir la distribucion asimetrica.</div>
    </div>
    <div class="pipeline-card">
      <div class="pipeline-num">04</div>
      <div class="pipeline-title">Split 80/20</div>
      <div class="pipeline-desc">Division antes del encoding para evitar data leakage y garantizar evaluacion real.</div>
    </div>
    <div class="pipeline-card">
      <div class="pipeline-num">05</div>
      <div class="pipeline-title">XGBoost</div>
      <div class="pipeline-desc">Gradient boosting secuencial. Cada arbol corrige los errores del anterior.</div>
    </div>
    <div class="pipeline-card">
      <div class="pipeline-num">06</div>
      <div class="pipeline-title">SHAP + LIME</div>
      <div class="pipeline-desc">Explicabilidad local y global. Cada prediccion se descompone en factores comprensibles.</div>
    </div>
  </div>
  <div class="modelo-cards">
    <div class="modelo-card">
      <div class="modelo-card-label">R² XGBoost</div>
      <div class="modelo-card-val">92,43%</div>
      <div class="modelo-card-sub">Varianza del precio explicada</div>
    </div>
    <div class="modelo-card">
      <div class="modelo-card-label">Error promedio (MAE)</div>
      <div class="modelo-card-val">$2,19M</div>
      <div class="modelo-card-sub">Pesos chilenos de error medio</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECCION 4 — RESULTADOS (imagenes estaticas)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="seccion">
  <div class="seccion-tag">Resultados</div>
  <h2 class="seccion-titulo">¿Que variables determinan el precio?</h2>
  <p class="seccion-desc">
    El analisis global SHAP sobre 300 vehiculos del conjunto de prueba revela
    que el modelo especifico, el año de fabricacion y el kilometraje son los
    tres factores con mayor incidencia en el precio, patron consistente entre
    Random Forest y XGBoost.
  </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**SHAP Global — Importancia y direccion**")
    st.image("xgb_shap_global_beeswarm.png",
             caption="Figura 1. Beeswarm SHAP XGBoost (300 vehiculos)",
             use_container_width=True)

with col2:
    st.markdown("**SHAP Global — Ranking de importancia**")
    st.image("xgb_shap_global_bar.png",
             caption="Figura 2. Importancia media |SHAP| XGBoost",
             use_container_width=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECCION 5 — SIMULADOR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="seccion">
  <div class="seccion-tag">Simulador</div>
  <h2 class="seccion-titulo">Tasa tu vehiculo</h2>
  <p class="seccion-desc">
    Ingresa las caracteristicas de tu vehiculo y el modelo XGBoost estimara
    su precio de mercado, explicando que factores determinaron esa estimacion
    mediante SHAP y LIME.
  </p>
</div>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        marca  = st.selectbox("🚗 Marca", MARCAS)
        modelo = st.selectbox("🔧 Modelo", CAT_MODELOS.get(marca, []))
        año    = st.number_input("📅 Año", min_value=1990,
                                 max_value=2025, value=2019, step=1)
        km     = st.number_input("📏 Kilometraje", min_value=0,
                                 max_value=500000, value=60000, step=5000)
        transmision = st.selectbox("⚙️ Transmision",  TRANSMISIONES)
        traccion    = st.selectbox("🔄 Traccion",     TRACCIONES)

    with col2:
        carroceria  = st.selectbox("🚙 Carroceria",  CARROCERIAS)
        combustible = st.selectbox("⛽ Combustible", COMBUSTIBLES)
        color       = st.selectbox("🎨 Color",       COLORES)
        paises_disp   = CAT_PAIS.get(marca, ["Otros/desconocido"])
        pais          = st.selectbox("🌍 Pais origen", paises_disp)
        regiones_disp = CAT_REGION.get(pais, ["No aplica"])
        region        = st.selectbox("📍 Region comercial", regiones_disp)
        n_feat        = st.slider("Variables a mostrar", 6, 12, 10)

    tasar = st.button("🔍 Estimar precio y explicar")

# ── Resultado ─────────────────────────────────────────────────
if tasar:
    gama = GAMA.get(marca, "Media")
    fila = {
        "marca":               marca,
        "modelo_limpio":       modelo,
        "transmision_final":   transmision,
        "traccion_final":      traccion,
        "carroceria_extraida": carroceria,
        "combustible":         combustible,
        "color":               color,
        "pais_origen":         pais,
        "region_comercial":    region,
        "gama_marca":          gama,
    }
    vec = [float(año), float(km)]
    for c in FEATS:
        vec.append(float(ENC[c].get(fila[c], GM)))
    X = np.array(vec).reshape(1, -1)

    pred_log = float(xgb.predict(X)[0])
    pred_clp = np.expm1(pred_log)
    rmin     = pred_clp * 0.90
    rmax     = pred_clp * 1.10

    st.markdown(f"""
    <div class="price-card">
      <p class="price-label">Precio estimado</p>
      <p class="price-value">${pred_clp:,.0f}</p>
      <p class="price-sub">CLP &nbsp;·&nbsp;
         {marca} {modelo} {año} &nbsp;·&nbsp; {km:,} km</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"📊 Rango estimado: ${rmin:,.0f} — ${rmax:,.0f} CLP")

    vu = {
        "año": str(año), "kilometraje": f"{km:,} km",
        "marca": marca, "modelo_limpio": modelo,
        "transmision_final": transmision,
        "traccion_final": traccion,
        "carroceria_extraida": carroceria,
        "combustible": combustible,
        "color": color,
        "pais_origen": pais,
        "region_comercial": region,
        "gama_marca": gama,
    }

    col_s, col_l = st.columns(2)

    with col_s:
        st.markdown("#### 📊 SHAP Local")
        with st.spinner("Calculando SHAP..."):
            ev   = exp_shap.expected_value
            base = float(ev[0]) if hasattr(ev, "__len__") else float(ev)
            sv_raw = exp_shap.shap_values(X)
            sv = sv_raw[0][0] if isinstance(sv_raw, list) else (
                 sv_raw[0] if sv_raw.ndim == 2 else sv_raw)

            s   = dict(zip(ALL_FEATURES, shap_a_clp(sv, base)))
            top = sorted(s.items(), key=lambda x: abs(x[1]),
                         reverse=True)[:n_feat]
            etq = [f'{f}  [{vu.get(f,"")}]' for f, _ in top]
            efs = [v for _, v in top]

        st.pyplot(grafico(efs, etq,
            f"SHAP — {marca} {modelo} {año}\n${pred_clp:,.0f} CLP"))

    with col_l:
        st.markdown("#### 🔍 LIME Local")
        with st.spinner("Calculando LIME (~15 s)..."):
            lime_exp = exp_lime.explain_instance(
                data_row    = X[0],
                predict_fn  = xgb.predict,
                num_features= n_feat,
                num_samples = 500,
            )
            lista = sorted(lime_exp.as_list(),
                           key=lambda x: abs(x[1]), reverse=True)[:n_feat]
            pesos   = np.array([i[1] for i in lista])
            ef_lime = lime_a_clp(pesos, pred_log)

            def limpiar(raw):
                for f in sorted(ALL_FEATURES, key=len, reverse=True):
                    if f in raw:
                        return f'{f}  [{vu.get(f,"")}]'
                return re.sub(r'\b(\d+)\.0+\b', r'\1', raw).strip()

            etq_lime = [limpiar(i[0]) for i in lista]

        st.pyplot(grafico(ef_lime, etq_lime,
            f"LIME — {marca} {modelo} {año}\n${pred_clp:,.0f} CLP"))

    st.caption(
        "Estimacion referencial basada en datos ChileAutos 2024. "
        "No constituye una tasacion oficial."
    )

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  <strong style="color:#4B5580">AutoTasar Chile</strong><br>
  Investigacion academica · Magister en Negocios Digitales<br>
  Universidad Diego Portales · Santiago, Chile · 2025<br><br>
  <span style="color:#2D3A55">
  Lundberg & Lee (2017) · Ribeiro et al. (2016) · Chen & Guestrin (2016) ·
  Nandan & Ghosh (2023) · Bergmann & Feuerriegel (2025)
  </span>
</div>
""", unsafe_allow_html=True)
