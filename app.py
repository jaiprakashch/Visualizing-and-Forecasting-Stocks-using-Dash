import dash
from dash import dcc
from dash import html
from datetime import datetime as dt
import yfinance as yf
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
# model
from model import lstm_prediction
from sklearn.svm import SVR


def get_stock_price_fig(df):

    fig = px.line(df,
                  x="Date",
                  y=["Close", "Open"],
                  title="Closing and Openning Price vs Date")

    return fig

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df,
                     x="Date",
                     y="EWA_20",
                     title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig

# Your existing functions and app initialization code remain the same

app = dash.Dash(
    __name__,
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Roboto&display=swap"]
)

# Adjusted layout with CSS for static and scrollable sections
app.layout = html.Div([
    html.Div([
        # Static Navigation
        html.P(" Stock VizFor ", className="start"),
        html.Div([
            html.P("Input stock code: "),
            html.Div([
                dcc.Input(id="dropdown_tickers", type="text"),
                html.Button("", id='submit'),
            ], className="form")
        ], className="input-place"),
        html.Div([
            dcc.DatePickerRange(id='my-date-picker-range',
                                min_date_allowed=dt(1995, 8, 5),
                                max_date_allowed=dt.now(),
                                initial_visible_month=dt.now(),
                                end_date=dt.now().date()),
        ], className="date"),
        html.Div([
            html.Button("Stock Price", className="stock-btn", id="stock"),
            html.Button("Indicators", className="indicators-btn", id="indicators"),
            dcc.Input(id="n_days", type="text", placeholder="number of days"),
            html.Button("Forecast", className="forecast-btn", id="forecast")
        ], className="buttons"),
    ], className="nav", style={'flex': '0 0 350px', 'height': '100vh', 'overflowY': 'hidden'}),
    
    html.Div([
        # Scrollable Content
         html.Div(
                [  # header
                    html.Img(id="logo",src="", style={'width': '100%', 'height': 'auto'}),
                    html.P(id="ticker")
                ]),
        html.Div(id="description", className="description_ticker"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content"),
    ], className="content", style={'flex': '1', 'overflowY': 'auto', 'height': '100vh'}),
], className="container", style={'display': 'flex', 'height': '100vh'})


# Your callback functions and app.run_server call remain unchange

# callback for company info
@app.callback([
    Output("description", "children"),
    Output("logo", "src"),
    Output("ticker", "children"),
    Output("stock", "n_clicks"),
    Output("indicators", "n_clicks"),
    Output("forecast", "n_clicks")
], [Input("submit", "n_clicks")], [State("dropdown_tickers", "value")])
def update_data(n, val):  # inpur parameter(s)
    if n == None:
        return "Hey there! Please enter a legitimate stock code to get details.", "https://wallpaperaccess.com/full/1393720.jpg", "Stocks", None, None , None
        # raise PreventUpdate
    else:
        if val == None:
            raise PreventUpdate
        else:
           ticker = yf.Ticker(val)
           inf = ticker.info
           df = pd.DataFrame().from_dict(inf, orient="index").T
           df = df[['logo_url', 'shortName', 'longBusinessSummary']]
           if 'logo_url' in df and 'shortName' in df and 'longBusinessSummary' in df:
             logo_url = df['logo_url'].values[0]
             shortName = df['shortName'].values[0]
             longBusinessSummary = df['longBusinessSummary'].values[0]
           else:
            # Set default values or handle the situation where one or more columns are missing
             logo_url = 'Default Logo URL'
             shortName = 'Unknown Company Name'
             longBusinessSummary = 'No description available'

           #logo_url = df['logo_url'].values[0] if 'logo_url' in df else 'Default Logo URL'
           #shortName = df['shortName'].values[0] if 'shortName' in df else 'Unknown Company Name'
           #longBusinessSummary = df['longBusinessSummary'].values[0] if 'longBusinessSummary' in df else 'No description available'
            
           return logo_url, longBusinessSummary, shortName, None, None, None



# callback for stocks graphs  

# Assuming you have an existing callback that fetches stock data
# and you want to include multiple visualizations in the response

@app.callback([
    Output("graphs-content", "children"),
], [
    Input("stock", "n_clicks"),  # Assuming "stock" is your button ID
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def update_stock_visualizations(n, start_date, end_date, val):
    if n is None or val is None:
        raise PreventUpdate

    # Fetch stock data
    df = yf.download(val, start_date, end_date)
    df.reset_index(inplace=True)

    # Visualization 1: Line chart for closing and opening prices
    fig = get_stock_price_fig(df)

    # Visualization 2: Candlestick chart for detailed price movement
    candlestick_fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                                     open=df['Open'], high=df['High'],
                                                     low=df['Low'], close=df['Close'])])
    candlestick_fig.update_layout(title='Stock Price Movement', xaxis_rangeslider_visible=False)

    # Return both figures wrapped in dcc.Graph components
    return [html.Div([
        dcc.Graph(figure=fig),
        dcc.Graph(figure=candlestick_fig)
    ])]

# callback for indicators
@app.callback([Output("main-content", "children")], [
    Input("indicators", "n_clicks"),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date')
], [State("dropdown_tickers", "value")])
def indicators(n, start_date, end_date, val):
    if n == None:
        return [""]
    if val == None:
        return [""]

    if start_date == None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


# callback for forecast
@app.callback([Output("forecast-content", "children")],
              [Input("forecast", "n_clicks")],
              [State("n_days", "value"),
               State("dropdown_tickers", "value")])
def forecast(n, n_days, val):
    if n == None:
        return [""]
    if val == None:
        raise PreventUpdate
    fig = lstm_prediction(val, int(n_days) + 1)
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run_server(debug=True)