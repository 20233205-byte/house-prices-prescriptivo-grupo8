# =====================================================================
# HOUSE PRICES – APP UNIFICADA (OLS, LASSO, RIDGE Y RANDOM FOREST)
# Modelamiento Predictivo de Datos · Universidad de Lima
#
# Ejecutar en Anaconda Prompt (en la carpeta donde está este archivo):
#   pip install -r requirements.txt
#   streamlit run app.py
# Si train.csv está en la misma carpeta, se carga solo; si no, súbelo
# desde la barra lateral.
# =====================================================================
import os
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------------------
st.set_page_config(page_title="House Prices – Comparación de modelos", page_icon="🏠", layout="wide")

TARGET = "SalePrice"
TEST_SIZE = 0.20          # 80 % entrenamiento / 20 % prueba (igual que el informe)
RS = 42                   # random_state común a todas las técnicas
TEAL, TEAL_CLARO, NARANJA, GRIS, FONDO = "#1F5A6B", "#8FB3BD", "#E07B39", "#5B6770", "#EEF3F7"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})

MODELOS = ["OLS", "Lasso", "Ridge", "Random Forest"]


def metricas(y_real, y_pred):
    return {
        "R2": r2_score(y_real, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "MAE": mean_absolute_error(y_real, y_pred),
    }


def fmt_usd(x):
    return f"${x:,.0f}"


def grafico_real_vs_predicho(y_real, y_pred, titulo, color=TEAL):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_real, y_pred, alpha=0.55, s=22, color=color, edgecolor="none")
    lo = min(np.min(y_real), np.min(y_pred))
    hi = max(np.max(y_real), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color=NARANJA, lw=2, label="Predicción perfecta")
    ax.set_xlabel("Precio real (USD)")
    ax.set_ylabel("Precio predicho (USD)")
    ax.set_title(titulo, color=TEAL)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def grafico_residuos(y_real, y_pred, titulo="Distribución de residuos"):
    fig, ax = plt.subplots(figsize=(12, 3.6))
    sns.histplot(np.asarray(y_real) - np.asarray(y_pred), kde=True, color=TEAL, ax=ax)
    ax.axvline(0, linestyle="--", color="black")
    ax.set_xlabel("Residuo = Real − Predicho (USD)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(titulo, color=TEAL)
    fig.tight_layout()
    return fig


def grafico_coeficientes(nombres, valores, titulo, xlabel="Coeficiente estandarizado"):
    fig, ax = plt.subplots(figsize=(7, 6))
    colores = [TEAL if v >= 0 else NARANJA for v in valores]
    ax.barh(nombres, valores, color=colores)
    ax.axvline(0, color=GRIS, lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, color=TEAL)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------------------------------
st.sidebar.title("🏠 House Prices")
st.sidebar.caption("Modelamiento Predictivo de Datos · Universidad de Lima")
st.sidebar.header("1. Carga de datos")
archivo = st.sidebar.file_uploader("Sube train.csv (Kaggle House Prices)", type=["csv", "xlsx"])


@st.cache_data
def leer_archivo(contenido, nombre):
    if nombre.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(contenido))
    return pd.read_excel(BytesIO(contenido))


@st.cache_data
def leer_local(ruta):
    return pd.read_csv(ruta)


RUTA_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")
if archivo is not None:
    df = leer_archivo(archivo.getvalue(), archivo.name)
elif os.path.exists(RUTA_LOCAL):
    df = leer_local(RUTA_LOCAL)
    st.sidebar.success("Se cargó train.csv de la carpeta de la app.")
else:
    st.title("🏠 Predicción del precio de viviendas")
    st.info("👈 Sube el archivo train.csv en la barra lateral para comenzar "
            "(o colócalo en la misma carpeta que app.py).")
    st.stop()

if TARGET not in df.columns:
    st.error("El dataset debe contener la variable SalePrice.")
    st.stop()

st.sidebar.info(f"Partición común: 80 % entrenamiento / 20 % prueba · random_state = {RS}")

# ---------------------------------------------------------------------
# ENTRENAMIENTO DE LOS CUATRO MODELOS (mismo procedimiento que cada notebook)
# ---------------------------------------------------------------------
OLS_PREDICTORES = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea",
                   "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd",
                   "YearBuilt", "YearRemodAdd", "Neighborhood", "KitchenQual", "ExterQual"]


@st.cache_resource(show_spinner="Entrenando OLS...")
def entrenar_ols(df):
    df_clean = df[[TARGET] + OLS_PREDICTORES].dropna().copy()
    y_raw = df_clean[TARGET]
    X_raw = df_clean[OLS_PREDICTORES]
    cat_cols = X_raw.select_dtypes(include=["object", "category"]).columns.tolist()
    X_enc = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_raw, test_size=TEST_SIZE, random_state=RS)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LinearRegression().fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    coef_df = pd.DataFrame({"Variable": X_enc.columns, "Coeficiente_Estandarizado": model.coef_})
    coef_df["Abs_Coef"] = coef_df["Coeficiente_Estandarizado"].abs()
    coef_df = coef_df.sort_values("Abs_Coef", ascending=False)
    return dict(df_clean=df_clean, cat_cols=cat_cols, X_test=X_test, y_test=y_test, y_pred=y_pred,
                model=model, coef_df=coef_df, nombres=X_enc.columns.tolist(), **metricas(y_test, y_pred))


@st.cache_resource(show_spinner="Entrenando Lasso (validación cruzada)...")
def entrenar_lasso(df):
    X = df.drop(columns=[TARGET, "Id"], errors="ignore").copy()
    y = df[TARGET]
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        if X[col].isna().any():
            moda = X[col].mode()
            X[col] = X[col].fillna(moda.iloc[0] if len(moda) > 0 else "Missing")
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RS)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lasso_cv = LassoCV(alphas=np.logspace(-3, 5, 150), cv=5, max_iter=20000, random_state=RS)
    lasso_cv.fit(X_train_s, y_train)
    model = Lasso(alpha=lasso_cv.alpha_, max_iter=20000).fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    coef_df = pd.DataFrame({"Variable": X.columns, "Coeficiente": model.coef_})
    coef_df["Abs_Coeficiente"] = coef_df["Coeficiente"].abs()
    coef_df = coef_df.sort_values("Abs_Coeficiente", ascending=False)
    return dict(alpha=lasso_cv.alpha_, n_num=len(num_cols), n_cat=len(cat_cols), n_train=len(X_train),
                X_test=X_test, y_test=y_test, y_pred=y_pred, coef_df=coef_df, **metricas(y_test, y_pred))


@st.cache_resource(show_spinner="Entrenando Ridge (validación cruzada)...")
def entrenar_ridge(df):
    df_clean = df.drop(columns=[c for c in ["Id", TARGET] if c in df.columns]).copy()
    y = df[TARGET]
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    nulos_antes = (int(df_clean[num_cols].isnull().sum().sum()), int(df_clean[cat_cols].isnull().sum().sum()))
    df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
    df_clean[cat_cols] = df_clean[cat_cols].fillna("Sin_Dato")
    X = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RS)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5).fit(X_train_s, y_train)
    y_pred = ridge_cv.predict(X_test_s)
    coef = pd.Series(ridge_cv.coef_, index=X.columns)
    return dict(alpha=ridge_cv.alpha_, nulos_antes=nulos_antes, n_cols=X.shape[1],
                n_orig=len(num_cols) + len(cat_cols), X_test=X_test, y_test=y_test, y_pred=y_pred,
                coef=coef, **metricas(y_test, y_pred))


RF_MEJORES = dict(n_estimators=400, max_depth=20, max_features=1.0, min_samples_leaf=2)


@st.cache_resource(show_spinner="Entrenando Random Forest...")
def entrenar_rf(df, optimizar=False):
    X = df.drop(columns=[TARGET, "Id"], errors="ignore")
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RS)
    X_train, X_test = X_train.copy(), X_test.copy()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    medianas_train = X_train[num_cols].median()       # solo con train: sin fuga de datos
    X_train[cat_cols] = X_train[cat_cols].fillna("Sin_Dato")
    X_train[num_cols] = X_train[num_cols].fillna(medianas_train)
    X_test[cat_cols] = X_test[cat_cols].fillna("Sin_Dato")
    X_test[num_cols] = X_test[num_cols].fillna(medianas_train)
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    if optimizar:   # misma búsqueda del notebook
        param_dist = {"n_estimators": [200, 400], "max_depth": [None, 20],
                      "max_features": [1.0, "sqrt"], "min_samples_leaf": [1, 2]}
        search = RandomizedSearchCV(RandomForestRegressor(random_state=RS), param_distributions=param_dist,
                                    n_iter=6, cv=3, scoring="r2", random_state=RS, n_jobs=-1)
        search.fit(X_train_enc, y_train)
        model, params = search.best_estimator_, search.best_params_
    else:
        model = RandomForestRegressor(random_state=RS, n_jobs=-1, **RF_MEJORES).fit(X_train_enc, y_train)
        params = dict(RF_MEJORES)
    y_pred = model.predict(X_test_enc)
    importance_df = pd.DataFrame({"Variable": X_train_enc.columns, "Importancia": model.feature_importances_}) \
        .sort_values("Importancia", ascending=False)
    return dict(model=model, params=params, columnas=X_train_enc.columns, cat_cols=cat_cols, num_cols=num_cols,
                medianas=medianas_train, X_test=X_test, y_test=y_test, y_pred=y_pred,
                importance_df=importance_df, train_idx=X_train.index, **metricas(y_test, y_pred))


st.sidebar.header("2. Random Forest")
optimizar_rf = st.sidebar.checkbox("Volver a optimizar hiperparámetros (RandomizedSearchCV, cv = 3)", value=False,
                                   help="Repite la búsqueda del notebook. Tarda unos minutos; "
                                        "sin marcar se usan los mejores hiperparámetros ya encontrados.")

ols = entrenar_ols(df)
lasso = entrenar_lasso(df)
ridge = entrenar_ridge(df)
rf = entrenar_rf(df, optimizar_rf)
RES = {"OLS": ols, "Lasso": lasso, "Ridge": ridge, "Random Forest": rf}

# ---------------------------------------------------------------------
# NAVEGACIÓN
# ---------------------------------------------------------------------
st.sidebar.header("3. Secciones")
pagina = st.sidebar.radio("Ir a:", [
    "Inicio",
    "Reducción de dimensiones",
    "OLS",
    "Lasso",
    "Ridge",
    "Random Forest",
    "Comparación y modelo seleccionado",
    "Escenarios (Random Forest)",
])

# =====================================================================
# 1. INICIO
# =====================================================================
if pagina == "Inicio":
    st.title("🏠 Predicción inteligente de precios de viviendas")
    st.subheader("Análisis comparativo de OLS, Lasso, Ridge y Random Forest")
    st.write("Estimar el precio de una vivienda es complejo: depende de su tamaño, calidad, antigüedad y "
             "ubicación. Esta aplicación compara cuatro técnicas de regresión sobre el mismo conjunto de prueba, "
             "selecciona la más precisa y la usa para evaluar tres escenarios de mejora inmobiliaria.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Viviendas", f"{df.shape[0]:,}")
    c2.metric("Variables", f"{df.shape[1]}")
    c3.metric("Precio medio", fmt_usd(df[TARGET].mean()))
    c4.metric("Precio mediano", fmt_usd(df[TARGET].median()))

    mejor = max(MODELOS, key=lambda m: RES[m]["R2"])
    st.success(f"**Modelo seleccionado: {mejor}** · R² = {RES[mejor]['R2']:.4f} · "
               f"RMSE = {fmt_usd(RES[mejor]['RMSE'])} · MAE = {fmt_usd(RES[mejor]['MAE'])}")

    with st.expander("Ver primeras filas del dataset"):
        st.dataframe(df.head(10))

    st.header("Análisis descriptivo")
    variables_clave = [v for v in ["SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF",
                                   "GarageArea", "YearBuilt", "LotArea", "1stFlrSF"] if v in df.columns]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estadística descriptiva")
        desc = df[variables_clave].describe().T
        desc["moda"] = df[variables_clave].mode().iloc[0]
        st.dataframe(desc[["mean", "moda", "std", "min", "max"]].round(2))
    with col2:
        st.subheader("Matriz de correlación")
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        sns.heatmap(df[variables_clave].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# =====================================================================
# 2. REDUCCIÓN DE DIMENSIONES
# =====================================================================
elif pagina == "Reducción de dimensiones":
    st.title("🧮 Análisis de reducción de dimensiones")
    st.write("Se evalúa si los siete predictores numéricos clave pueden resumirse en menos dimensiones. "
             "Todo se calcula con el conjunto de entrenamiento (80 %) después de winsorizar las variables "
             "de superficie entre los percentiles 1 y 99, para evitar fuga de información.")

    pred7 = ["GrLivArea", "OverallQual", "TotalBsmtSF", "GarageArea", "YearBuilt", "LotArea", "1stFlrSF"]
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=RS)
    Xtr = tr[pred7].copy()
    Xte = te[pred7].copy()
    for c in ["GrLivArea", "TotalBsmtSF", "1stFlrSF", "LotArea"]:
        p1, p99 = Xtr[c].quantile([0.01, 0.99])
        Xtr[c] = Xtr[c].clip(p1, p99)
        Xte[c] = Xte[c].clip(p1, p99)

    # VIF = diagonal de la inversa de la matriz de correlación
    R = np.corrcoef(Xtr.values, rowvar=False)
    R_inv = np.linalg.inv(R)
    vif = pd.Series(np.diag(R_inv), index=pred7).sort_values(ascending=False)
    # Bartlett y KMO
    n, p = Xtr.shape
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    gl = p * (p - 1) / 2
    p_valor = stats.chi2.sf(chi2, gl)
    parcial = -R_inv / np.sqrt(np.outer(np.diag(R_inv), np.diag(R_inv)))
    np.fill_diagonal(parcial, 0)
    R0 = R.copy()
    np.fill_diagonal(R0, 0)
    kmo = (R0 ** 2).sum() / ((R0 ** 2).sum() + (parcial ** 2).sum())
    msa = (R0 ** 2).sum(axis=0) / ((R0 ** 2).sum(axis=0) + (parcial ** 2).sum(axis=0))

    st.header("1. Diagnóstico de redundancia (VIF)")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.dataframe(pd.DataFrame({"VIF": vif.round(3), "Tolerancia (1/VIF)": (1 / vif).round(3),
                                   "R²_j": (1 - 1 / vif).round(3)}))
    with c2:
        st.write(f"Todos los VIF están entre **{vif.min():.2f}** y **{vif.max():.2f}**, por debajo de 5: "
                 "no hay multicolinealidad severa. Los más altos son 1stFlrSF y TotalBsmtSF, que tienen una "
                 "correlación de 0.82 (el sótano suele ocupar la misma huella que el primer piso).")
        st.metric("Índice de condición", f"{np.sqrt(np.linalg.eigvalsh(R).max() / np.linalg.eigvalsh(R).min()):.2f}",
                  help="Menor a 30: sin multicolinealidad grave.")

    st.header("2. Adecuación de los datos: Bartlett y KMO")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bartlett χ²", f"{chi2:,.2f}", help=f"gl = {gl:.0f}")
    c2.metric("p-valor", "< 0.001" if p_valor < 0.001 else f"{p_valor:.4f}")
    c3.metric("KMO global", f"{kmo:.3f}")
    st.write(f"MSA por variable: entre **{msa.min():.3f}** y **{msa.max():.3f}**. "
             "Las correlaciones son significativas y la adecuación es aceptable: procede aplicar PCA.")

    st.header("3. Análisis de Componentes Principales (PCA)")
    scaler7 = StandardScaler().fit(Xtr)
    Ztr, Zte = scaler7.transform(Xtr), scaler7.transform(Xte)
    pca = PCA().fit(Ztr)
    evr = pca.explained_variance_ratio_
    eig = evr * p
    cps = [f"CP{i + 1}" for i in range(p)]
    tabla_pca = pd.DataFrame({"Autovalor (λ)": eig.round(3), "% varianza": (evr * 100).round(2),
                              "% acumulado": (np.cumsum(evr) * 100).round(2),
                              "¿λ > 1?": np.where(eig > 1, "Sí", "No")}, index=cps)
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.dataframe(tabla_pca)
    with c2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].bar(range(1, p + 1), evr, color=TEAL)
        ax[0].axhline(1 / p, ls="--", color=NARANJA, lw=1)
        ax[0].set_title("Varianza explicada", color=TEAL)
        ax[0].set_xlabel("Componente")
        ax[1].plot(range(1, p + 1), np.cumsum(evr), "o-", color=TEAL)
        ax[1].axhline(0.9, ls="--", color=NARANJA, lw=1)
        ax[1].set_title("Varianza acumulada", color=TEAL)
        ax[1].set_xlabel("Componentes")
        fig.tight_layout()
        st.pyplot(fig)
    k90 = int(np.argmax(np.cumsum(evr) >= 0.90)) + 1
    st.write(f"Criterio de Kaiser (λ > 1): **{int((eig > 1).sum())} componentes** "
             f"({np.cumsum(evr)[int((eig > 1).sum()) - 1] * 100:.1f} %). "
             f"Varianza acumulada ≥ 90 %: **{k90} componentes** ({np.cumsum(evr)[k90 - 1] * 100:.1f} %).")

    st.subheader("Cargas de las variables (CP1 a CP5)")
    cargas = pd.DataFrame(pca.components_.T * np.sqrt(eig), index=pred7, columns=cps).iloc[:, :5]
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(cargas, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)
    st.write("**CP1** resume el tamaño y la calidad general (todas las cargas positivas); "
             "**CP2** contrapone antigüedad y extensión del terreno; **CP3**, la distribución del área "
             "(una planta frente a dos plantas).")

    st.subheader("Efecto de la reducción en la predicción (regresión sobre k componentes)")
    Ttr, Tte = pca.transform(Ztr), pca.transform(Zte)
    filas = []
    for k in range(1, p + 1):
        m = LinearRegression().fit(Ttr[:, :k], tr[TARGET])
        filas.append({"k componentes": k, "% var. acumulada": round(np.cumsum(evr)[k - 1] * 100, 2),
                      **{key: (round(v, 4) if key == "R2" else round(v)) for key, v in
                         metricas(te[TARGET], m.predict(Tte[:, :k])).items()}})
    st.dataframe(pd.DataFrame(filas), hide_index=True)
    st.info("Con 5 componentes se conserva más del 90 % de la información y el R² casi no cambia. "
            "Como la multicolinealidad es moderada, los modelos usan las variables originales y la "
            "redundancia se controla con Lasso y Ridge.")

# =====================================================================
# 3. OLS
# =====================================================================
elif pagina == "OLS":
    st.title("📐 Regresión OLS (Mínimos Cuadrados Ordinarios)")
    st.write("Línea base sin regularización: carga de datos, selección de variables, One-Hot Encoding, "
             "estandarización, entrenamiento OLS, métricas, ecuación y diagnóstico visual.")

    st.header("1. Variables y preprocesamiento")
    st.write("**Variables independientes (X):** " + ", ".join(OLS_PREDICTORES))
    st.info("Variables categóricas detectadas: " + ", ".join(ols["cat_cols"]) +
            ". Se aplicó One-Hot Encoding con drop_first=True "
            f"({len(ols['nombres'])} predictores después del encoding).")

    st.header("2. Métricas de desempeño")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²", f"{ols['R2']:.4f}")
    c2.metric("RMSE", f"{ols['RMSE']:,.2f}")
    c3.metric("MAE", f"{ols['MAE']:,.2f}")
    c4.metric("Observaciones de prueba", f"{len(ols['y_test']):,}")

    st.header("3. Ecuación del modelo OLS")
    terminos = [f"{ols['model'].intercept_:,.2f}"]
    for var, coef in zip(ols["nombres"], ols["model"].coef_):
        terminos.append(f"{'+' if coef >= 0 else '-'} {abs(coef):,.2f}·{var}*")
    st.code("SalePrice = " + " ".join(terminos))
    st.caption("* Los predictores de esta ecuación están estandarizados mediante Z-Score.")

    st.header("4. Diagnóstico visual")
    col_a, col_b = st.columns(2)
    with col_a:
        top = ols["coef_df"].head(15).sort_values("Coeficiente_Estandarizado")
        st.pyplot(grafico_coeficientes(top["Variable"], top["Coeficiente_Estandarizado"],
                                       "Coeficientes estandarizados (top 15)"))
    with col_b:
        st.pyplot(grafico_real_vs_predicho(ols["y_test"], ols["y_pred"], "Valores reales vs. predichos"))
    st.pyplot(grafico_residuos(ols["y_test"], ols["y_pred"]))

    st.header("5. Descarga de resultados")
    pred_df = ols["df_clean"].loc[ols["X_test"].index, OLS_PREDICTORES].copy()
    pred_df["Real"] = ols["y_test"]
    pred_df["Predicho"] = ols["y_pred"]
    pred_df["Residuo"] = pred_df["Real"] - pred_df["Predicho"]
    pred_df["Error_Absoluto"] = pred_df["Residuo"].abs()
    pred_df["Error_Porcentual"] = pred_df["Error_Absoluto"] / pred_df["Real"] * 100
    pred_df = pred_df.reset_index().rename(columns={"index": "ID"})
    metrics_df = pd.DataFrame({"Metrica": ["R2", "RMSE", "MAE", "N_Prueba"],
                               "Valor": [ols["R2"], ols["RMSE"], ols["MAE"], len(ols["y_test"])]})
    d1, d2, d3 = st.columns(3)
    d1.download_button("Descargar predicciones para Power BI", pred_df.to_csv(index=False).encode("utf-8"),
                       "predicciones_df.csv", "text/csv")
    d2.download_button("Descargar métricas", metrics_df.to_csv(index=False).encode("utf-8"),
                       "metrics_df.csv", "text/csv")
    d3.download_button("Descargar coeficientes", ols["coef_df"].to_csv(index=False).encode("utf-8"),
                       "coef_df.csv", "text/csv")

# =====================================================================
# 4. LASSO
# =====================================================================
elif pagina == "Lasso":
    st.title("📊 Regresión LASSO")
    st.write("Regresión LASSO con validación cruzada (5 pliegues) para seleccionar automáticamente el "
             "parámetro alpha. La penalización L1 lleva a cero los coeficientes de las variables poco relevantes.")

    st.header("1. Preparación de los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables numéricas", lasso["n_num"])
    c2.metric("Variables categóricas", lasso["n_cat"])
    c3.metric("Datos de entrenamiento", f"{lasso['n_train']:,}")

    st.header("2. Selección del parámetro Alpha")
    st.success(f"Alpha (λ) óptimo seleccionado: **{lasso['alpha']:.4f}**")

    st.header("3. Evaluación del modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{lasso['R2']:.4f}")
    c2.metric("RMSE", f"{lasso['RMSE']:,.2f}")
    c3.metric("MAE", f"{lasso['MAE']:,.2f}")
    st.write("**R²:** proporción de la variabilidad explicada por el modelo.")
    st.write("**RMSE:** tamaño de los errores, dando mayor peso a errores grandes.")
    st.write("**MAE:** promedio de las diferencias absolutas entre valores reales y predichos.")

    st.header("4. Análisis de coeficientes")
    coef_df = lasso["coef_df"]
    ceros = int((coef_df["Coeficiente"] == 0).sum())
    activos = len(coef_df) - ceros
    c1, c2 = st.columns(2)
    c1.metric("Variables totales", len(coef_df))
    c2.metric("Variables con coeficiente 0", ceros)
    st.write(f"LASSO mantuvo **{activos} variables** con coeficientes diferentes de cero.")
    with st.expander("Ver tabla completa de coeficientes"):
        st.dataframe(coef_df)
    col_a, col_b = st.columns(2)
    with col_a:
        top = coef_df.head(15).sort_values("Coeficiente")
        st.pyplot(grafico_coeficientes(top["Variable"], top["Coeficiente"], "Principales coeficientes del modelo LASSO"))
    with col_b:
        st.pyplot(grafico_real_vs_predicho(lasso["y_test"], lasso["y_pred"], "Valores reales vs. predichos"))
    st.pyplot(grafico_residuos(lasso["y_test"], lasso["y_pred"], "Distribución de los residuos"))

    st.header("5. Resultados de predicción")
    st.dataframe(pd.DataFrame({"Valor_Real": lasso["y_test"].values, "Valor_Predicho": lasso["y_pred"],
                               "Residuo": lasso["y_test"].values - lasso["y_pred"]}).head(20))

    st.header("Conclusión")
    st.write(f"El modelo de Regresión LASSO obtuvo un R² de **{lasso['R2']:.4f}**, un RMSE de "
             f"**{lasso['RMSE']:,.2f}** y un MAE de **{lasso['MAE']:,.2f}** sobre el conjunto de prueba. "
             f"El parámetro alpha seleccionado mediante validación cruzada fue **{lasso['alpha']:.4f}**. "
             f"La regularización llevó {ceros} coeficientes a cero, realizando una selección automática de variables.")

# =====================================================================
# 5. RIDGE
# =====================================================================
elif pagina == "Ridge":
    st.title("🏠 Regresión Ridge")
    st.caption("Dimensión de reducción de varianza (regularización L2)")

    st.header("1. Preprocesamiento")
    st.write(f"Nulos antes de imputar → numéricas: {ridge['nulos_antes'][0]}, categóricas: {ridge['nulos_antes'][1]}. "
             "Numéricas imputadas con la mediana; categóricas como “Sin_Dato”.")
    st.write(f"Total de columnas después de codificar: **{ridge['n_cols']}** (de {ridge['n_orig']} variables originales).")

    st.header("2. Resultados del modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mejor alpha", f"{ridge['alpha']:.2f}")
    m2.metric("R²", f"{ridge['R2']:.4f}")
    m3.metric("RMSE", fmt_usd(ridge["RMSE"]))
    m4.metric("MAE", fmt_usd(ridge["MAE"]))

    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(grafico_real_vs_predicho(ridge["y_test"], ridge["y_pred"], "Real vs. predicho"))
    with col4:
        top15 = ridge["coef"].abs().sort_values(ascending=False).head(15).index
        serie = ridge["coef"][top15].sort_values()
        st.pyplot(grafico_coeficientes(serie.index, serie.values, "Top 15 variables más influyentes"))

    if "Neighborhood" in df.columns:
        st.header("3. Explorar por barrio")
        barrios_test = df.loc[ridge["y_test"].index, "Neighborhood"]
        barrio_sel = st.selectbox("Selecciona un barrio", ["Todos"] + sorted(barrios_test.unique()))
        if barrio_sel != "Todos":
            mask = (barrios_test == barrio_sel).values
            st.write(f"Mostrando {mask.sum()} viviendas del barrio **{barrio_sel}**")
            st.pyplot(grafico_real_vs_predicho(ridge["y_test"][mask], ridge["y_pred"][mask],
                                               f"Barrio {barrio_sel}", color=NARANJA))

# =====================================================================
# 6. RANDOM FOREST
# =====================================================================
elif pagina == "Random Forest":
    st.title("🏡 Predicción de precios de vivienda con Random Forest")
    st.write("Imputación con datos de entrenamiento (mediana y “Sin_Dato”), One-Hot Encoding alineado entre "
             "entrenamiento y prueba, y un **Random Forest Regressor** con los hiperparámetros encontrados "
             "mediante **RandomizedSearchCV**.")

    st.header("1. Métricas de desempeño")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Coeficiente R²", f"{rf['R2']:.4f}")
    k2.metric("RMSE (Error Cuadrático)", f"{rf['RMSE']:,.2f}")
    k3.metric("MAE (Error Absoluto)", f"{rf['MAE']:,.2f}")
    k4.metric("Árboles (n_estimators)", f"{rf['params'].get('n_estimators', '-')}")
    with st.expander("⚙️ Ver hiperparámetros utilizados por el modelo"):
        st.json(rf["params"])

    st.header("2. Importancia de variables")
    st.dataframe(rf["importance_df"].head(15), hide_index=True)

    st.header("3. Diagnóstico visual")
    c1, c2 = st.columns(2)
    with c1:
        top = rf["importance_df"].head(15).sort_values("Importancia")
        st.pyplot(grafico_coeficientes(top["Variable"], top["Importancia"],
                                       "Variables con mayor influencia en la predicción", "Importancia"))
    with c2:
        st.pyplot(grafico_real_vs_predicho(rf["y_test"], rf["y_pred"], "Ajuste del modelo (Random Forest)"))
    st.pyplot(grafico_residuos(rf["y_test"], rf["y_pred"], "Verificación de la distribución de residuos"))

    st.header("4. Tabla de resultados")
    resultado = rf["X_test"].copy()
    resultado["SalePrice_real"] = rf["y_test"]
    resultado["SalePrice_predicho"] = rf["y_pred"]
    resultado["Diferencia"] = resultado["SalePrice_real"] - resultado["SalePrice_predicho"]
    resultado["Diferencia_abs"] = resultado["Diferencia"].abs()
    st.dataframe(resultado.head(20))
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        resultado.to_excel(writer, sheet_name="Predicciones", index=False)
        pd.DataFrame({"Métrica": ["RMSE", "MAE", "R²"], "Resultado": [rf["RMSE"], rf["MAE"], rf["R2"]]}) \
            .to_excel(writer, sheet_name="Metricas", index=False)
        rf["importance_df"].head(15).to_excel(writer, sheet_name="Importancia_Variables", index=False)
    st.download_button("📥 Descargar resultados en Excel", buffer.getvalue(), "resultados_random_forest.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# =====================================================================
# 7. COMPARACIÓN Y MODELO SELECCIONADO
# =====================================================================
elif pagina == "Comparación y modelo seleccionado":
    st.title("🏆 Comparación de modelos y modelo seleccionado")
    st.write("Las cuatro técnicas se evalúan con R², RMSE y MAE sobre el mismo conjunto de prueba "
             f"({len(rf['y_test'])} viviendas).")

    tabla = pd.DataFrame({m: {"R²": RES[m]["R2"], "RMSE (USD)": RES[m]["RMSE"], "MAE (USD)": RES[m]["MAE"]}
                          for m in MODELOS}).T
    mejor = tabla["R²"].idxmax()
    st.dataframe(tabla.style.format({"R²": "{:.4f}", "RMSE (USD)": "{:,.0f}", "MAE (USD)": "{:,.0f}"})
                 .highlight_max(subset=["R²"], color="#CFE3E8")
                 .highlight_min(subset=["RMSE (USD)", "MAE (USD)"], color="#CFE3E8"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.2]})
    yy = np.arange(len(MODELOS))[::-1]
    for yv, m in zip(yy, MODELOS):
        col = TEAL if m == mejor else TEAL_CLARO
        ax[0].hlines(yv, 0.80, RES[m]["R2"], color=col, lw=3)
        ax[0].plot(RES[m]["R2"], yv, "o", ms=12, color=col)
        ax[0].text(RES[m]["R2"] + 0.004, yv, f"{RES[m]['R2']:.3f}", va="center", fontweight="bold")
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels(MODELOS)
    ax[0].set_xlim(0.80, 0.91)
    ax[0].set_title("R² (más alto = mejor)", color=TEAL)
    x = np.arange(len(MODELOS))
    ax[1].bar(x - 0.19, [RES[m]["RMSE"] / 1000 for m in MODELOS], 0.38, color=TEAL, label="RMSE")
    ax[1].bar(x + 0.19, [RES[m]["MAE"] / 1000 for m in MODELOS], 0.38, color=NARANJA, label="MAE")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(MODELOS)
    ax[1].set_ylabel("Miles de USD")
    ax[1].set_title("RMSE y MAE (más bajo = mejor)", color=TEAL)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig)

    st.header(f"Modelo seleccionado: {mejor}")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{RES[mejor]['R2']:.3f}", help="Proporción de la variación del precio explicada")
    c2.metric("RMSE", fmt_usd(RES[mejor]["RMSE"]))
    c3.metric("MAE", fmt_usd(RES[mejor]["MAE"]),
              f"{RES[mejor]['MAE'] / rf['y_test'].mean() * 100:.1f} % del precio medio", delta_color="off")

    st.subheader("Reducción del error frente a los modelos lineales")
    mejora = pd.DataFrame({m: {"RMSE": f"−{(1 - RES[mejor]['RMSE'] / RES[m]['RMSE']) * 100:.1f} %",
                               "MAE": f"−{(1 - RES[mejor]['MAE'] / RES[m]['MAE']) * 100:.1f} %"}
                           for m in MODELOS if m != mejor}).T
    st.dataframe(mejora)
    err = np.abs(rf["y_pred"] - rf["y_test"].values) / rf["y_test"].values
    st.write(f"El **{(err <= 0.20).mean() * 100:.0f} %** de las predicciones queda a menos de 20 % del precio real "
             f"y el **{(err <= 0.10).mean() * 100:.0f} %** a menos de 10 %.")

    st.subheader("Importancia de variables (Random Forest)")
    top10 = rf["importance_df"].head(10).sort_values("Importancia")
    st.pyplot(grafico_coeficientes(top10["Variable"], top10["Importancia"], "Top 10 variables", "Importancia relativa"))
    st.info(f"Calidad general (OverallQual) y superficie habitable (GrLivArea) concentran el "
            f"{rf['importance_df'].head(2)['Importancia'].sum() * 100:.0f} % de la importancia. "
            "Por eso Random Forest se usa para evaluar los escenarios de mejora.")

# =====================================================================
# 8. ESCENARIOS (RANDOM FOREST)
# =====================================================================
elif pagina == "Escenarios (Random Forest)":
    st.title("🏡 Escenarios prescriptivos con Random Forest")
    st.write("Se toma una vivienda real del dataset (ID 834) y se estima cuánto cambiaría su precio con tres "
             "mejoras. Cada escenario parte de la misma vivienda original y usa el modelo ya entrenado.")

    def preparar(fila):
        Xf = fila.drop(columns=[TARGET, "Id"], errors="ignore").copy()
        Xf[rf["cat_cols"]] = Xf[rf["cat_cols"]].fillna("Sin_Dato")
        Xf[rf["num_cols"]] = Xf[rf["num_cols"]].fillna(rf["medianas"])
        Xf = pd.get_dummies(Xf, columns=rf["cat_cols"])
        return Xf.reindex(columns=rf["columnas"], fill_value=0)

    def predecir(fila):
        return float(rf["model"].predict(preparar(fila))[0])

    ESCENARIOS = {
        "1. Remodelación": {"OverallQual": 7, "OverallCond": 7, "KitchenQual": "Gd", "ExterQual": "Gd",
                            "YearRemodAdd": 2009},
        "2. Ampliación": {"GrLivArea": 1744, "1stFlrSF": 1744, "TotRmsAbvGrd": 7, "YearRemodAdd": 2009},
        "3. Mejora integral": {"OverallQual": 8, "OverallCond": 8, "KitchenQual": "Gd", "ExterQual": "Gd",
                               "GrLivArea": 1744, "1stFlrSF": 1744, "GarageArea": 566, "TotRmsAbvGrd": 7,
                               "YearRemodAdd": 2009},
    }
    VARS_ESC = ["OverallQual", "OverallCond", "KitchenQual", "ExterQual", "GrLivArea", "1stFlrSF",
                "GarageArea", "TotRmsAbvGrd", "YearRemodAdd"]

    # Costos de obra: Remodeling, Cost vs. Value 2010–11 (promedios nacionales de EE. UU., en USD).
    # La ampliación (sala de 16 × 25 ft = 400 ft²) y el garaje (26 × 26 ft = 676 ft²) se prorratean por ft².
    COSTO_UNIT = {"Cocina (remodelación menor)": 21695, "Cocina (remodelación mayor)": 58367,
                  "Fachada (revestimiento de vinilo)": 11357, "Ventanas (vinilo)": 11066,
                  "Baño (remodelación)": 16634, "Ampliación (USD por ft²)": 85740 / 400,
                  "Garaje (USD por ft²)": 60608 / 676}
    PARTIDAS = {"1. Remodelación": ["Cocina (remodelación menor)", "Fachada (revestimiento de vinilo)",
                                    "Ventanas (vinilo)", "Baño (remodelación)"],
                "2. Ampliación": [],
                "3. Mejora integral": ["Cocina (remodelación mayor)", "Fachada (revestimiento de vinilo)",
                                       "Ventanas (vinilo)", "Baño (remodelación)"]}

    if 834 not in set(df["Id"]):
        st.error("No se encontró la vivienda ID 834 en el dataset cargado.")
        st.stop()
    base = df[df["Id"] == 834].copy()

    with st.expander("Supuestos de costo de las obras (puedes editarlos)"):
        st.caption("Fuente: Remodeling, Cost vs. Value 2010–11, promedios nacionales de EE. UU. (USD).")
        cu = {k: st.number_input(k, min_value=0.0, value=float(v), step=100.0,
                                 format="%.2f" if "ft²" in k else "%.0f") for k, v in COSTO_UNIT.items()}

    def costo_obra(nombre, cambios):
        """Devuelve las partidas de costo (USD) de un escenario."""
        partidas = {p: cu[p] for p in PARTIDAS[nombre]}
        ft2_ampl = cambios.get("GrLivArea", base["GrLivArea"].iloc[0]) - base["GrLivArea"].iloc[0]
        ft2_gar = cambios.get("GarageArea", base["GarageArea"].iloc[0]) - base["GarageArea"].iloc[0]
        if ft2_ampl > 0:
            partidas[f"Ampliación de {ft2_ampl:.0f} ft²"] = ft2_ampl * cu["Ampliación (USD por ft²)"]
        if ft2_gar > 0:
            partidas[f"Garaje +{ft2_gar:.0f} ft²"] = ft2_gar * cu["Garaje (USD por ft²)"]
        return partidas

    st.header("1. Cambios de cada escenario")
    tabla_cambios = pd.DataFrame({"Base": base[VARS_ESC].iloc[0]})
    for nombre, cambios in ESCENARIOS.items():
        tabla_cambios[nombre] = [cambios.get(v, base[v].iloc[0]) for v in VARS_ESC]
    st.dataframe(tabla_cambios.astype(str))
    st.caption(f"Vivienda ID 834: barrio {base['Neighborhood'].iloc[0]}, construida en {base['YearBuilt'].iloc[0]}, "
               f"precio real {fmt_usd(base[TARGET].iloc[0])}. TA = típica · Gd = buena.")

    st.header("2. Resultados")
    precio_base = predecir(base)
    filas = [{"Escenario": "Base (ID 834)", "Precio estimado": precio_base, "Incremento ($)": 0.0, "Incremento (%)": 0.0,
              "Costo de la obra ($)": 0.0, "Ganancia neta ($)": 0.0, "Recupera (%)": np.nan}]
    detalle_costos = []
    for nombre, cambios in ESCENARIOS.items():
        esc = base.copy()
        for var, val in cambios.items():
            esc[var] = val
        precio = predecir(esc)
        incremento = precio - precio_base
        partidas = costo_obra(nombre, cambios)
        costo = sum(partidas.values())
        detalle_costos += [{"Escenario": nombre, "Obra": p, "Costo ($)": c} for p, c in partidas.items()]
        filas.append({"Escenario": nombre, "Precio estimado": precio, "Incremento ($)": incremento,
                      "Incremento (%)": incremento / precio_base * 100, "Costo de la obra ($)": costo,
                      "Ganancia neta ($)": incremento - costo, "Recupera (%)": incremento / costo * 100})
    tabla_esc = pd.DataFrame(filas)

    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.dataframe(tabla_esc[["Escenario", "Precio estimado", "Incremento ($)", "Incremento (%)"]]
                     .style.format({"Precio estimado": "${:,.0f}", "Incremento ($)": "${:,.0f}",
                                    "Incremento (%)": "{:+.1f} %"}),
                     hide_index=True)
        mejor_esc = tabla_esc.iloc[1:]["Incremento (%)"].idxmax()
        st.metric("Mayor aumento", tabla_esc.loc[mejor_esc, "Escenario"],
                  f"+{tabla_esc.loc[mejor_esc, 'Incremento (%)']:.1f} %")
    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        etiquetas = ["Base", "Esc. 1\nRemodelación", "Esc. 2\nAmpliación", "Esc. 3\nMejora integral"]
        valores = tabla_esc["Precio estimado"].values / 1000
        pcts = tabla_esc["Incremento (%)"].values
        barras = ax.bar(etiquetas, valores, color=[TEAL_CLARO, TEAL, TEAL, NARANJA], width=0.64)
        for b, v, pc in zip(barras, valores, pcts):
            xc = b.get_x() + b.get_width() / 2
            ax.text(xc, v - 8, f"${v:,.1f} k", ha="center", va="top", color="white", fontweight="bold")
            if pc > 0:
                ax.text(xc, v + 4, f"+{pc:.1f} %", ha="center", va="bottom", fontweight="bold",
                        color=NARANJA if pc == pcts.max() else TEAL)
        ax.axhline(base[TARGET].iloc[0] / 1000, ls="--", color=GRIS, lw=1.2)
        ax.set_ylim(0, valores.max() * 1.18)
        ax.set_ylabel("Precio estimado (miles USD)")
        ax.set_title("Escenarios · vivienda ID 834", color=TEAL)
        fig.tight_layout()
        st.pyplot(fig)

    st.header("3. Rentabilidad: aumento del precio frente al costo de la obra")
    st.write("**Recupera (%)** = aumento del precio / costo de la obra × 100. "
             "**Ganancia neta** = aumento del precio − costo de la obra.")
    r1, r2 = st.columns([1, 1.1])
    with r1:
        st.dataframe(tabla_esc.iloc[1:][["Escenario", "Incremento ($)", "Costo de la obra ($)", "Ganancia neta ($)",
                                         "Recupera (%)"]]
                     .style.format({"Incremento ($)": "${:,.0f}", "Costo de la obra ($)": "${:,.0f}",
                                    "Ganancia neta ($)": "${:,.0f}", "Recupera (%)": "{:.1f} %"}),
                     hide_index=True)
        mejor_rec = tabla_esc.iloc[1:]["Recupera (%)"].idxmax()
        st.metric("Mejor recuperación de la inversión", tabla_esc.loc[mejor_rec, "Escenario"],
                  f"{tabla_esc.loc[mejor_rec, 'Recupera (%)']:.0f} % del costo")
        with st.expander("Detalle de costos por obra"):
            st.dataframe(pd.DataFrame(detalle_costos).style.format({"Costo ($)": "${:,.0f}"}), hide_index=True)
    with r2:
        esc_r = tabla_esc.iloc[1:]
        x = np.arange(len(esc_r))
        inc_k, cos_k = esc_r["Incremento ($)"].values / 1000, esc_r["Costo de la obra ($)"].values / 1000
        fig, ax = plt.subplots(figsize=(6.5, 4.6))
        b1 = ax.bar(x - 0.18, inc_k, 0.36, color=TEAL, label="Aumento del precio")
        b2 = ax.bar(x + 0.18, cos_k, 0.36, color=NARANJA, label="Costo de la obra")
        for barras_, col in ((b1, TEAL), (b2, NARANJA)):
            for b in barras_:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"${b.get_height():,.1f} k",
                        ha="center", va="bottom", fontsize=9, color=col, fontweight="bold")
        for xi, c, rc in zip(x, cos_k, esc_r["Recupera (%)"].values):
            ax.text(xi, c + 17, f"recupera {rc:.0f} %", ha="center", va="bottom", fontweight="bold",
                    color=TEAL if rc == esc_r["Recupera (%)"].max() else GRIS)
        ax.set_xticks(x)
        ax.set_xticklabels(["Esc. 1\nRemodelación", "Esc. 2\nAmpliación", "Esc. 3\nMejora integral"])
        ax.set_ylim(0, max(cos_k.max(), inc_k.max()) * 1.26)
        ax.set_ylabel("Miles de USD")
        ax.set_title("Aumento del precio vs. costo de la obra", color=TEAL)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig)

    st.header("4. Efecto de cada cambio por separado")
    efectos = []
    for nombre, cambios in ESCENARIOS.items():
        for var, val in cambios.items():
            esc = base.copy()
            esc[var] = val
            efectos.append({"Escenario": nombre, "Variable": var, "Nuevo valor": str(val),
                            "Efecto aislado ($)": round(predecir(esc) - precio_base)})
    st.dataframe(pd.DataFrame(efectos), hide_index=True)
    st.caption("En Random Forest los efectos no se suman exactamente: las variables interactúan entre sí.")

    rec = dict(zip(tabla_esc["Escenario"], tabla_esc["Recupera (%)"]))
    st.success(f"**Recomendación:** con estos costos, ninguna obra se paga sola con la venta. La remodelación es la "
               f"mejor opción: recupera el {rec['1. Remodelación']:.0f} % de su costo, frente al "
               f"{rec['3. Mejora integral']:.0f} % de la mejora integral y solo el {rec['2. Ampliación']:.0f} % de la "
               "ampliación. Conviene priorizar la calidad y los acabados; ampliar no se justifica.")
    st.warning("Supuestos de la propuesta grupal. Costos: Remodeling, Cost vs. Value 2010–11 (promedios de EE. UU.); "
               "la ampliación y el garaje se prorratean por ft². La vivienda ID 834 pertenece al conjunto de "
               "entrenamiento y la predicción no implica causalidad.")

    # ---------------- Simulador ----------------
    st.header("5. Simulador: arma tu propio escenario")
    ids = df["Id"].tolist()
    id_sel = st.selectbox("Vivienda base (Id)", ids, index=ids.index(834))
    casa = df[df["Id"] == id_sel].copy()
    precio_casa = predecir(casa)
    calidades = ["Po", "Fa", "TA", "Gd", "Ex"]

    s1, s2, s3 = st.columns(3)
    with s1:
        oq = st.slider("Calidad general (OverallQual)", 1, 10, int(casa["OverallQual"].iloc[0]))
        oc = st.slider("Estado de conservación (OverallCond)", 1, 9, int(casa["OverallCond"].iloc[0]))
        kq = st.select_slider("Calidad de cocina (KitchenQual)", calidades,
                              value=casa["KitchenQual"].iloc[0] if casa["KitchenQual"].iloc[0] in calidades else "TA")
    with s2:
        gla = st.number_input("Superficie habitable, ft² (GrLivArea)", 300, 6000, int(casa["GrLivArea"].iloc[0]), 10)
        f1 = st.number_input("Área del primer piso, ft² (1stFlrSF)", 300, 5000, int(casa["1stFlrSF"].iloc[0]), 10)
        ga = st.number_input("Área del garaje, ft² (GarageArea)", 0, 1500, int(casa["GarageArea"].iloc[0]), 10)
    with s3:
        eq = st.select_slider("Calidad exterior (ExterQual)", calidades,
                              value=casa["ExterQual"].iloc[0] if casa["ExterQual"].iloc[0] in calidades else "TA")
        tr_ = st.slider("Ambientes sobre el suelo (TotRmsAbvGrd)", 2, 14, int(casa["TotRmsAbvGrd"].iloc[0]))
        yr = st.slider("Año de remodelación (YearRemodAdd)", int(casa["YearBuilt"].iloc[0]), 2010,
                       int(casa["YearRemodAdd"].iloc[0]))

    sim = casa.copy()
    for var, val in {"OverallQual": oq, "OverallCond": oc, "KitchenQual": kq, "ExterQual": eq, "GrLivArea": gla,
                     "1stFlrSF": f1, "GarageArea": ga, "TotRmsAbvGrd": tr_, "YearRemodAdd": yr}.items():
        sim[var] = val
    precio_sim = predecir(sim)
    m1, m2, m3 = st.columns(3)
    m1.metric("Precio real", fmt_usd(casa[TARGET].iloc[0]))
    m2.metric("Precio estimado (sin cambios)", fmt_usd(precio_casa))
    m3.metric("Precio estimado con tus cambios", fmt_usd(precio_sim),
              f"{(precio_sim - precio_casa) / precio_casa * 100:+.1f} %")

st.sidebar.markdown("---")
st.sidebar.caption("Proyecto EC · House Prices · Universidad de Lima")
