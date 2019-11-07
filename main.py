from flask import Flask, render_template, flash, request, url_for, Markup
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging, io, base64, os, datetime
from datetime import datetime
from datetime import timedelta
from lxml import html
import time, os, string, requests
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROLLING_PERIOD = 5
PREDICT_OUT_PERIOD = 5
SAVED_DNN_MODEL_PATH = 'DNNRegressors'
FEATURES = [str(id) for id in range(0,ROLLING_PERIOD)]

# global variables
stock_market_historical_data = None
stock_market_live_data = None
options_stocks = None

# load nasdaq corollary material
stock_company_info_amex = None
stock_company_info_nasdaq = None
stock_company_info_nyse = None
reloaded_model = None

def load_fundamental_company_info():
    global stock_company_info_amex, stock_company_info_nasdaq, stock_company_info_nyse
    import pandas as pd
    stock_company_info_amex = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_amex.csv'))
    stock_company_info_nasdaq = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_nasdaq.csv'))
    stock_company_info_nyse = pd.read_csv(os.path.join(BASE_DIR, 'stock_company_info_nyse.csv'))

def get_fundamental_information(symbol):
    CompanyName = "No company name"
    Sector = "No sector"
    Industry = "No industry"
    MarketCap = "No market cap"
    Exchange = 'No exchange'

    if (symbol in list(stock_company_info_nasdaq['Symbol'])):
        data_row = stock_company_info_nasdaq[stock_company_info_nasdaq['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'NASDAQ'

    elif (symbol in list(stock_company_info_amex['Symbol'])):
        data_row = stock_company_info_amex[stock_company_info_amex['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'AMEX'

    elif (symbol in list(stock_company_info_nyse['Symbol'])):
        data_row = stock_company_info_nyse[stock_company_info_nyse['Symbol'] == symbol]
        CompanyName = data_row['Name'].values[0]
        Sector = data_row['Sector'].values[0]
        Industry = data_row['industry'].values[0]
        MarketCap = data_row['MarketCap'].values[0]
        Exchange = 'NYSE'

    return (CompanyName, Sector, Industry, MarketCap, Exchange)

# !sudo pip3 install wikipedia
def get_wikipedia_intro(symbol):
    import wikipedia
    company_name = get_fundamental_information(symbol)[0]
    description = wikipedia.page(company_name).content
    return(description.split('\n')[0])

def get_stock_prediction(symbol):
    def get_input_live_fn(data_set, num_epochs=None, shuffle=True):
        return tf.estimator.inputs.pandas_input_fn(
          x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
          num_epochs=num_epochs,
          shuffle=shuffle)

    temp_df = stock_market_live_data[stock_market_live_data['symbol']==symbol]
    pred_gen = reloaded_model.predict(input_fn=get_input_live_fn(temp_df, num_epochs=1, shuffle=False))

    predictions = []
    for p in (list(pred_gen)):
        predictions.append(p['predictions'][0])

    # call this to stop growing the graph and save memory
    tf.compat.v1.get_default_graph().finalize()

    return(predictions)

def get_plot_prediction(symbol):

    predictions = get_stock_prediction(symbol)

    if (len(predictions) > 0):
        temp_df = stock_market_live_data[stock_market_live_data['symbol']==symbol]

        actuals = list(temp_df[FEATURES].values[0])
        # transform log price to price of past data
        actuals = list(np.exp(actuals))

        days_before = temp_df['last_market_date'].values[-1]
        days_before_list = []
        for d in range(ROLLING_PERIOD):
            days_before_list.append((np.busday_offset(np.datetime64(days_before,'D'),-d, roll='backward')))
        days_before_list = sorted(days_before_list)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(days_before_list, actuals, color='green', linewidth=4)

        for d in range(1, PREDICT_OUT_PERIOD+1):
            days_before_list.append((np.busday_offset(np.datetime64(days_before,'D'),d, roll='forward')))
            actuals.append(np.exp(predictions[0]))

        days_before_list = sorted(days_before_list)
        plt.suptitle('Forecast for ' + str(temp_df['date'].values[-1])[0:10] + ': $' +
                     str(np.round(np.exp(predictions[0]),2)))

        ax.plot(days_before_list, actuals, color='blue', linestyle='dashed')
        ax.grid()

        plt.xticks(days_before_list, days_before_list, fontsize = 7)
        ax.set_xticklabels(days_before_list, rotation = 35, ha="right")


        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_bit_to_text = base64.b64encode(img.getvalue()).decode()

        chart_plot = Markup('<img style="padding:1px; border:1px solid #021a40; width: 300px; height: 400px" src="data:image/png;base64,{}">'.format(plot_bit_to_text))
        return(chart_plot)

@app.before_first_request
def prepare_data():
    global stock_market_historical_data, stock_market_live_data, options_stocks, reloaded_model
    stock_market_historical_data = pd.read_csv(os.path.join(BASE_DIR, 'stock_market_historical_data.csv'))
    stock_market_live_data = pd.read_csv(os.path.join(BASE_DIR, 'stock_market_live_data.csv'))
    options_stocks = sorted(set(stock_market_historical_data['symbol']))
    load_fundamental_company_info()

    # restor tensorflow model
    FEATURES = ['0', '1', '2', '3', '4']
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]
    reloaded_model = tf.estimator.DNNRegressor(
        feature_columns=feature_cols,
        hidden_units=[5, 5, 5],
        model_dir=(os.path.join(BASE_DIR, 'DNNRegressors/')))

    # call this to stop growing the graph and save memory
    tf.compat.v1.get_default_graph().finalize()

# Define a route for the default URL, which loads the form
@app.route('/', methods=['POST', 'GET'])
def get_financial_information():
    chart_plot = ''
    fundamentals_company_name = ''
    fundamentals_sector = ''
    fundamentals_industry = ''
    fundamentals_marketcap = ''
    fundamentals_exchange = ''

    wiki = ''
    selected_stock = ''

    if request.method == 'POST':
        selected_stock = request.form['selected_stock']
        fundamentals = get_fundamental_information(selected_stock)

        if not  fundamentals[0] is np.NaN:
            fundamentals_company_name = Markup("<b>" + str(fundamentals[0]) + "</b><BR><BR>")
        if not fundamentals[1] is np.NaN:
            fundamentals_sector = Markup("Sector: <b>" + str(fundamentals[1]) + "</b><BR><BR>")
        if not fundamentals[2] is np.NaN:
            fundamentals_industry = Markup("Industry: <b>" + str(fundamentals[2]) + "</b><BR><BR>")
        if not fundamentals[3] is np.NaN:
            fundamentals_marketcap = Markup("MarketCap: <b>$" + str(fundamentals[3]) + "</b><BR><BR>")
        if not fundamentals[4] is np.NaN:
            fundamentals_exchange = Markup("Exchange: <b>" + str(fundamentals[4]) + "</b><BR><BR>")

        wiki = get_wikipedia_intro(selected_stock)
        chart_plot = get_plot_prediction(selected_stock)

    return render_template('stock-market-report.html',
        options_stocks=options_stocks,
        selected_stock = selected_stock,
        chart_plot=chart_plot,
        wiki = wiki,
        fundamentals_company_name = fundamentals_company_name,
        fundamentals_sector = fundamentals_sector,
        fundamentals_industry = fundamentals_industry,
        fundamentals_marketcap = fundamentals_marketcap,
        fundamentals_exchange = fundamentals_exchange)


if __name__ == "__main__":
    app.run(debug=True)