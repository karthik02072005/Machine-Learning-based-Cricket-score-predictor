import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import joblib
import os

# --- SETUP ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load('ipl_score_model.pkl')
df = pd.read_csv('ipl_ml_ready.csv')

teams = sorted(df['batting_team'].unique())
venues = sorted(df['venue'].unique())
venue_stats = df.groupby('venue')['final_score'].mean().to_dict()

# 🎨 PROFESSIONAL TEAM COLOR MAPPING
team_colors = {
    "Mumbai Indians": "#004BA0", "Chennai Super Kings": "#FFFF00",
    "Royal Challengers Bengaluru": "#EC1C24", "Kolkata Knight Riders": "#2E0854",
    "Sunrisers Hyderabad": "#FF822E", "Delhi Capitals": "#00008B",
    "Rajasthan Royals": "#FF69B4", "Punjab Kings": "#D71920",
    "Gujarat Titans": "#1B2133", "Lucknow Super Giants": "#0057E7"
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, dbc.icons.BOOTSTRAP])

# --- GLASS STYLE ---
GLASS_STYLE = {
    "background": "rgba(255, 255, 255, 0.03)",
    "borderRadius": "20px",
    "border": "1px solid rgba(255, 255, 255, 0.1)",
    "backdropFilter": "blur(12px)",
    "padding": "25px",
    "boxShadow": "0 10px 40px rgba(0,0,0,0.8)"
}

def create_card(title, component_id, height="350px"):
    return html.Div([
        html.H6(title, className="text-uppercase mb-3", style={"color": "#00ffcc", "letterSpacing": "2px"}),
        dcc.Graph(id=component_id, config={'displayModeBar': False}, style={"height": height})
    ], style=GLASS_STYLE)

# --- LAYOUT ---
app.layout = html.Div([
    dbc.Container([
        html.H1("🏏 IPL STRATEGY ENGINE", className="text-center mt-4", 
                style={"color": "#f1c40f", "fontWeight": "900", "letterSpacing": "5px"}),
        
        html.H5(id="live-match-header", className="text-center text-info mb-5", style={"letterSpacing": "2px"}),

        dbc.Row([
            # LEFT INPUT PANEL
            dbc.Col([
                html.Div([
                    html.H5("CONFIGURATION", className="mb-4 text-warning"),
                    html.Label("Batting Team", className="text-muted small"),
                    dcc.Dropdown(id='batting-team', options=teams, value=teams[0], className="mb-3"),
                    html.Label("Bowling Team", className="text-muted small"),
                    dcc.Dropdown(id='bowling-team', options=teams, value=teams[1], className="mb-3"),
                    html.Label("Venue", className="text-muted small"),
                    dcc.Dropdown(id='venue', options=venues, value=venues[0], className="mb-4"),
                    
                    html.Hr(style={"borderColor": "rgba(255,255,255,0.1)"}),
                    
                    html.Label("Match Progress", className="text-info small"),
                    html.P(id='label-score', className="mb-0 mt-2 small"),
                    dcc.Slider(id='score', min=0, max=250, value=100, step=1),
                    html.P(id='label-overs', className="mb-0 mt-2 small"),
                    dcc.Slider(id='overs', min=5, max=19, value=10, step=0.1),
                    html.P(id='label-wickets', className="mb-0 mt-2 small"),
                    dcc.Slider(id='wickets', min=0, max=9, value=2, step=1),
                ], style=GLASS_STYLE)
            ], md=4),

            # RIGHT VISUAL PANEL
            dbc.Col([
                dbc.Row([
                    dbc.Col(create_card("PROJECTED SCORE", "score-gauge"), md=6),
                    dbc.Col(create_card("RUN RATE METRICS", "rr-metrics"), md=6),
                ], className="mb-4"),
                dbc.Col(create_card("GROUND PAR COMPARISON", "venue-bar", height="250px"), width=12),
                
                html.Div(id="insight-box", style={
                    "marginTop": "20px", "padding": "20px", "borderRadius": "15px",
                    "background": "rgba(0,0,0,0.5)", "borderLeft": "5px solid #f1c40f", "color": "white"
                })
            ], md=8)
        ])
    ], fluid=True)
], style={"background": "linear-gradient(135deg, #050505, #102027)", "minHeight": "100vh", "padding": "20px"})

# --- CALLBACKS ---
@app.callback(
    [Output('score-gauge', 'figure'), Output('venue-bar', 'figure'), Output('rr-metrics', 'figure'),
     Output('label-score', 'children'), Output('label-overs', 'children'), Output('label-wickets', 'children'),
     Output('live-match-header', 'children'), Output('insight-box', 'children')],
    [Input('batting-team', 'value'), Input('bowling-team', 'value'), Input('venue', 'value'),
     Input('score', 'value'), Input('overs', 'value'), Input('wickets', 'value')]
)
def update_ui(bat, bowl, ven, score, overs, wickets):
    balls_left = 120 - (overs * 6)
    crr = round(score / overs, 2)
    input_df = pd.DataFrame([[bat, bowl, ven, score, balls_left, wickets, crr*5, 1]], 
                            columns=['batting_team', 'bowling_team', 'venue', 'current_score', 'balls_left', 'wickets_fallen', 'runs_last_5', 'wickets_last_5'])
    
    predicted = int(model.predict(input_df)[0])
    avg_v = int(venue_stats.get(ven, 170))
    team_color = team_colors.get(bat, "#00ffcc")

    # Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=predicted,
        number={'font': {'color': team_color, 'size': 50}},
        gauge={'axis': {'range': [None, 260]}, 'bar': {'color': team_color}, 'bgcolor': "rgba(0,0,0,0)"}
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, margin=dict(t=0, b=0))

    # Metrics
    fig_rr = go.Figure()
    fig_rr.add_trace(go.Indicator(mode="number+delta", value=crr, title={"text": "CURRENT RR"}, delta={'reference': 8.5}, domain={'x': [0, 1], 'y': [0.5, 1]}))
    fig_rr.add_trace(go.Indicator(mode="number", value=round((predicted-score)/(balls_left/6), 2), title={"text": "NEEDED RR"}, domain={'x': [0, 1], 'y': [0, 0.5]}))
    fig_rr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, margin=dict(t=20, b=20))

    # Venue Bar
    fig_venue = go.Figure(data=[
        go.Bar(name='PREDICTED', x=[ven], y=[predicted], marker_color=team_color, width=0.2),
        go.Bar(name='VENUE AVG', x=[ven], y=[avg_v], marker_color='rgba(255,255,255,0.2)', width=0.2)
    ])
    fig_venue.update_layout(barmode='group', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, margin=dict(t=0, b=0))

    insight = f"Based on historical data at {ven}, {bat} is on track for {predicted}. They need to maintain a rate of {round((predicted-score)/(balls_left/6), 2)} to hit this target."
    
    return fig_gauge, fig_venue, fig_rr, f"Runs: {score}", f"Overs: {overs}", f"Wickets: {wickets}", f"{bat} vs {bowl}", insight

if __name__ == '__main__':
    app.run(debug=True, port=8050)