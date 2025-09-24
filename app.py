app.layout = html.Div([

    # Encabezado
    html.Div([
        ...
    ], style={'padding': '10px 25px'}),

    # Controles
    html.Div([
        ...
    ], style={'padding': '10px 25px','borderBottom':'1px solid #ddd'}),

    # Métricas y conclusiones
    html.Div([
        ...
    ], style={'padding': '10px 25px'}),

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

    # Tabla de predicciones
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

    # Footer
    html.Div([
        html.P("Instrucciones: usa el selector de fechas para enfocar una ventana temporal; "
               "cambia el modelo para comparar métricas y predicciones. "
               "Este panel está pensado para apoyar decisiones de gestión hídrica en Suba.")
    ], style={'padding':'10px 25px','fontSize':'13px','color':'#666'})

], style={'fontFamily':'Arial, sans-serif'})
