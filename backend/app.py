from flask import Flask, request, render_template
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    file = request.files['datafile']
    data = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
    data = data.asfreq('D').fillna(0)

    model = ARIMA(data['Sales'], order=(5,1,0))
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=30)

    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['Sales'], label='Historical Sales')
    plt.plot(pd.date_range(start=data.index[-1], periods=31, freq='D')[1:], forecast, label='Forecast', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales Forecast')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

    return render_template('forecast.html', forecast_img=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
