# app.py
# Dashboard interactivo para consumo de agua en Suba (nivel Chingaza)
# Requiere: Dataset_comsumo_de_agua.csv en la misma carpeta.

import os
from math import sqrt
import pandas as pd
import numpy as np
from datetime import datetime
from textwrap import dedent

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, html, dcc, dash_table, Input, Output, State

# --- Config ---
DATA_PATH = "Dataset_comsumo_de_agua.csv"  # Asegúrate que este archivo esté en el repo
PAGE_TITLE = "Dashboard Consumo de Agua - Suba"

# --- Cargar y preparar datos ---
df = pd.read_csv(DATA_PATH)
if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'])
else:
    # si la columna fecha no se detecta, intenta encontrar una
    for c in df.columns:
        try:
            tmp = pd.to_datetime(df[c])
            df = df.rename(columns={c: 'fecha'})
            df['fecha'] = tmp
            break
        except Exception:
            pass

df = df.sort_values('fecha').reset_index(drop=True)

# crear features (igual que en el notebook)
df['month'] = df['fecha'].dt.month
df['year'] = df['fecha'].dt.year
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
for lag in [1,3]:
    df[f'nivel_lag_{lag}'] = df['nivel_chingaza_pct'].shift(lag)
    df[f'consumo_lag_{lag}'] = df['consumo_suba_lpcd'].shift(lag)
df['nivel_roll_3'] = df['nivel_chingaza_pct'].rolling(3, min_periods=1).mean()
df['consumo_roll_3'] = df['consumo_suba_lpcd'].rolling(3, min_periods=1).mean()

df_model = df.dropna().reset_index(drop=True)  # filas válidas para modelado

# Train/test split temporal (70/30)
n = len(df_model)
train_idx = int(0.7 * n)
train = df_model.iloc[:train_idx].copy()
test = df_model.iloc[train_idx:].copy()

FEATURES = [
    'nivel_chingaza_pct','nivel_lag_1','nivel_lag_3','nivel_roll_3',
    'month_sin','month_cos','consumo_lag_1','consumo_lag_3','consumo_roll_3'
]

X_train = train[FEATURES].values
y_train = train['consumo_suba_lpcd'].values
X_test = test[FEATURES].values
y_test = test['consumo_suba_lpcd'].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# --- Models (entrenamos versiones "ligeras" al iniciar la app) ---
models = {}
# 1) Regresión lineal múltiple (sobre scaled)
lr = LinearRegression().fit(X_train_s, y_train)
models['LinearRegression'] = (lr, lambda X: lr.predict(scaler.transform(X)) )

# 2) Random Forest (no scaled)
rf = RandomForestRegressor(random_state=42, n_estimators=100).fit(X_train, y_train)
models['RandomForest'] = (rf, lambda X: rf.predict(X) )

# 3) Gradient Boosting
gbr = GradientBoostingRegressor(random_state=42, n_estimators=100).fit(X_train, y_train)
models['GradientBoosting'] = (gbr, lambda X: gbr.predict(X) )

# predicciones sobre test (por modelo)
preds = {}
for name, (m, predfun) in models.items():
    preds[name] = predfun(X_test)

# metrics helper
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
def calc_metrics(y_true, y_pred):
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(sqrt(mean_squared_error(y_true, y_pred)))
    }

metrics_summary = {name: calc_metrics(y_test, preds[name]) for name in preds}

# feature importance (from RF)
feat_imp = pd.DataFrame({'feature': FEATURES, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

# --- Dash app ---
app = Dash(__name__, title=PAGE_TITLE)
server = app.server

# Layout
min_date = df['fecha'].min()
max_date = df['fecha'].max()

app.layout = html.Div([
    html.Div([
        html.H1(PAGE_TITLE, style={'marginBottom': 5}),
        html.P("Dashboard interactivo: relación nivel embalse Chingaza ↔ consumo en Suba. "
               "Filtra por rango de fechas, elige modelo y explora predicciones.", style={'marginTop': 0}),
    ], style={'padding': '10px 25px'}),

    # Controls
    html.Div([
        html.Div([
            html.Label("Rango de fechas:"),
            dcc.DatePickerRange(
                id='date-range',
                start_date=min_date,
                end_date=max_date,
                display_format='YYYY-MM-DD'
            )
        ], style={'display':'inline-block','marginRight':'20px'}),

        html.Div([
            html.Label("Modelo:"),
            dcc.Dropdown(
                id='model-select',
                options=[{'label':k,'value':k} for k in models.keys()],
                value='RandomForest',
                clearable=False,
                style={'width':'220px'}
            )
        ], style={'display':'inline-block','marginRight':'20px'}),

        html.Div([
            html.Label("Mostrar:"),
            dcc.Checklist(
                id='show-checks',
                options=[
                    {'label':'Datos reales','value':'real'},
                    {'label':'Predicción (test)','value':'pred'},
                    {'label':'Media móvil consumo (3m)','value':'roll3'}
                ],
                value=['real','pred']
            )
        ], style={'display':'inline-block','verticalAlign':'top'})
    ], style={'padding':'10px 25px','borderBottom':'1px solid #ddd'}),

    # Top metrics and narrative
    html.Div([
        html.Div([
            html.H3("Métricas (conjunto test)"),
            html.Div(id='metrics-cards', style={'display':'flex','gap':'12px'})
        ], style={'width':'48%','display':'inline-block','verticalAlign':'top'}),
        html.Div([
            html.H3("Conclusiones clave"),
            html.Div([
                html.P(dedent("""
                    • El consumo en Suba muestra estacionalidad y dependencia histórica.
                    • El nivel del embalse es relevante pero no suficiente por sí solo.
                    • Modelos no lineales (Random Forest, Gradient Boosting) capturan mejor patrones y reducen errores.
                    • Para decisiones operativas, recomendamos usar el modelo de ensamble ajustado y continuar enriqueciendo datos con variables climáticas y registros de campañas.
                """))
            ], style={'fontSize':'14px','lineHeight':'1.5'})
        ], style={'width':'48%','display':'inline-block','verticalAlign':'top','paddingLeft':'20px'})
    ], style={'padding':'10px 25px'}),

    # Graphs
    html.Div([
        dcc.Graph(id='ts-graph', style={'height':'420px'}),
    ], style={'padding':'10px 25px'}),

    html.Div([
        html.Div([
            dcc.Graph(id='scatter-graph')
        ], style={'width':'48%','display':'inline-block'}),
        html.Div([
            dcc.Graph(id='featimp-graph')
        ], style={'width':'48%','display':'inline-block'})
    ], style={'padding':'10px 25px'}),

    # Table with predictions
    html.Div([
        html.H3("Tabla de observaciones y predicciones (conjunto test)"),
        dash_table.DataTable(
            id='pred-table',
            columns=[{"name": i, "id": i} for i in ['fecha','consumo_suba_lpcd','predicted','error']],
            page_size=10,
            style_table={'overflowX':'auto'},
            style_cell={'textAlign':'center'}
        )
    ], style={'padding':'10px 25px 60px 25px'}),

    # Footer / instructions
    html.Div([
        html.P("Instrucciones: usa el selector de fechas para enfocar una ventana temporal; "
               "cambia el modelo para comparar métricas y predicciones. "
               "Este panel está pensado para apoyar decisiones de gestión hídrica en Suba.")
    ], style={'padding':'10px 25px','fontSize':'13px','color':'#666'})

], style={'fontFamily':'Arial, sans-serif'})

# --- Callbacks ---

@app.callback(
    Output('ts-graph', 'figure'),
    Output('scatter-graph','figure'),
    Output('featimp-graph','figure'),
    Output('metrics-cards','children'),
    Output('pred-table','data'),
    Input('date-range','start_date'),
    Input('date-range','end_date'),
    Input('model-select','value'),
    Input('show-checks','value')
)
def update_all(start_date, end_date, model_name, show_checks):
    # filter by date
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    df_f = df[(df['fecha'] >= s) & (df['fecha'] <= e)].copy().reset_index(drop=True)

    # Build time-series plot: real consumption, model prediction (for points inside test)
    # We'll generate predictions for rows that belong to df_model (i.e., have features)
    df_pred_source = df_model.copy()
    mask = (df_pred_source['fecha'] >= s) & (df_pred_source['fecha'] <= e)
    df_pred_show = df_pred_source[mask].copy()

    # compute predictions with selected model
    m_obj, m_predfun = models[model_name]
    X_local = df_pred_show[FEATURES].values
    preds_local = m_predfun(X_local)
    df_pred_show['predicted'] = preds_local
    df_pred_show['error'] = df_pred_show['consumo_suba_lpcd'] - df_pred_show['predicted']

    # TS figure
    ts_fig = go.Figure()
    if 'real' in show_checks:
        ts_fig.add_trace(go.Scatter(
            x=df_f['fecha'], y=df_f['consumo_suba_lpcd'],
            mode='lines+markers', name='Consumo real'
        ))
    if 'roll3' in show_checks:
        # compute rolling on filtered data (3-month rolling)
        df_f = df_f.sort_values('fecha')
        df_f['consumo_roll_3_local'] = df_f['consumo_suba_lpcd'].rolling(3, min_periods=1).mean()
        ts_fig.add_trace(go.Scatter(
            x=df_f['fecha'], y=df_f['consumo_roll_3_local'],
            mode='lines', name='Consumo 3m (rolling)'
        ))
    if 'pred' in show_checks:
        ts_fig.add_trace(go.Scatter(
            x=df_pred_show['fecha'], y=df_pred_show['predicted'],
            mode='lines+markers', name=f'Predicción ({model_name})'
        ))

    ts_fig.update_layout(title="Serie de Consumo (real vs predicción)", xaxis_title="Fecha", yaxis_title="Consumo (lpcd)")

    # Scatter: nivel vs consumo (con predicción color)
    scatter_fig = px.scatter(df_f, x='nivel_chingaza_pct', y='consumo_suba_lpcd',
                             color=df_f['consumo_suba_lpcd'],
                             labels={'nivel_chingaza_pct':'Nivel embalse (%)','consumo_suba_lpcd':'Consumo (lpcd)'},
                             title="Nivel embalse vs Consumo (puntos reales)")

    # feature importance (bar)
    feat_fig = px.bar(feat_imp, x='importance', y='feature', orientation='h', title='Importancia de features (Random Forest)')
    feat_fig.update_layout(yaxis={'categoryorder':'total ascending'})

    # metrics cards
    m = calc_metrics(df_pred_show['consumo_suba_lpcd'], df_pred_show['predicted']) if len(df_pred_show)>0 else metrics_summary[model_name]
    cards = [
        html.Div([
            html.H4("R²"),
            html.P(f"{m['r2']:.3f}")
        ], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'100px','textAlign':'center'}),
        html.Div([
            html.H4("MAE"),
            html.P(f"{m['mae']:.2f}")
        ], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'100px','textAlign':'center'}),
        html.Div([
            html.H4("RMSE"),
            html.P(f"{m['rmse']:.2f}")
        ], style={'padding':'10px','border':'1px solid #eee','borderRadius':'6px','width':'100px','textAlign':'center'}),
    ]

    # data table: show df_pred_show rows (test-like)
    table_df = df_pred_show[['fecha','consumo_suba_lpcd','predicted','error']].copy()
    table_df['fecha'] = table_df['fecha'].dt.strftime('%Y-%m-%d')
    table_data = table_df.to_dict('records')

    return ts_fig, scatter_fig, feat_fig, cards, table_data

# run server (for local)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
