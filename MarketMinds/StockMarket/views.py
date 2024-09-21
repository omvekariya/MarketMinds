# MarketMinds © 2024 by Om Vekariya
# Licensed under the MarketMinds Proprietary Software License
# Commercial use, personal use, modification, and redistribution are prohibited.

from django.shortcuts import render,HttpResponse
from django.http import JsonResponse
import requests
import yfinance as yf
import csv
import numpy as np
import pandas as pd
from datetime import datetime,timedelta, date
# Libraries for model
import tensorflow.keras as keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM

df=None
df1=None
df2=None

def company_search(request):
    return render(request, 'company_search.html')

def get_companies(request):
    company_name = request.GET.get('company_name')
    try:
        suggestions = get_company_suggestions(company_name)
    except Exception as e:
        return JsonResponse({"internetError" : "Please check internet connection"})
    if not suggestions:
        return JsonResponse({"error": "No company found for " + company_name}, status=404)
    
    return JsonResponse({"data" : suggestions})

def get_company_suggestions(query):
    url = f'https://query2.finance.yahoo.com/v1/finance/search?q={query}'
    response = requests.get(url, headers = {'User-agent': 'Super Bot Power Level Over 9000'})
    if response.status_code == 200:
        return response.json()['quotes']
    return []

def get_company(request):
    symbol = request.GET.get('symbol')
    company_info, company_news = fetch_company_details(symbol)
    return JsonResponse({
        "company_info": company_info[0],
        "company_news": company_news,
    })

def fetch_company_details(symbol):
    url = f'https://query2.finance.yahoo.com/v1/finance/search?q={symbol}'
    response = requests.get(url, headers = {'User-agent': 'Super Bot Power Level Over 9000'})
    if response.status_code == 200:
        return response.json()['quotes'], response.json()['news']
    return {}

def fetch_company_news(symbol):
    url = f'https://query2.finance.yahoo.com/v2/finance/news?symbols={symbol}'
    response = requests.get(url, headers = {'User-agent': 'Super Bot Power Level Over 9000'})
    if response.status_code == 200:
        return response.json().get('items', {}).get('result', [])
    return []

def index(request):
    return render(request, 'index.html')

def details(request):
    global df
    context={"flag":False}
    if request.method == 'POST':
        try:
            stocks = str(request.POST.get('company1'))
            start_date = str(request.POST.get('start_date'))
            close_date= str(request.POST.get('close_date'))
        

            comp = yf.Ticker(stocks)
            des=comp.info
            temp_des={}
            for key in des:
                if des[key]!='None' and des[key]!=[]:
                    temp_des[key]=des[key]
            des=temp_des
        
            df = yf.download(stocks, start_date, close_date)
            df['Date']=df.index.strftime('%d-%m-%Y')
            x=list(map(str,df.index.strftime('%Y-%m-%d')))
            
            y_high=list(df['High'])
            y_open=list(df['Open'])
            y_low=list(df['Low'])
            y_close=list(df['Close'])
            y_volume=list(df['Volume'])
            
        
            context={
                'x':x,
                'y_high':y_high,
                'y_low':y_low,
                'y_open':y_open,
                'y_close':y_close,
                'y_volume':y_volume,
                'df':df,
                'predicted_x':[1,2,3,4,5],
                'predicted_y':[5,4,3,2,1],
                'max_price':round(max(y_high),2),
                'min_price':round(min(y_low),2),
                'last_day_price':round(y_close[-1],2),
                'change_in_price':round(y_high[-1]-y_high[0],2),
                'change_in_precentage': round(((y_high[-1]-y_high[0])/y_high[0])*100,2),
                'change_color': 'red' if round(((y_high[-1]-y_high[0])/y_high[0])*100,2) < 0 else 'green',
                "description":des,
                "flag":True,
                'company':stocks,
                'start_date':start_date,
                'close_date':close_date
            }
        except Exception as e:
            context = {
                "flag": False,
                "err" : "Something went wrong while fetching data",
                "errSolution": "Please check internet connection or company code"
            }
    
    return render(request,'details.html',context)

def compare(request):
    context={
        "flag":False
    }
    if request.method == 'POST':
        try:
            stocks1 = request.POST.get('company1')
            stocks2 = request.POST.get('company2')
            start_date = str(request.POST.get('start_date'))
            close_date= str(request.POST.get('close_date'))
        
            global df1,df2

            df1 = yf.download(stocks1, start_date, close_date)

            df1['Date']=df1.index.strftime('%d-%m-%Y')
    
            x_stock1=list(map(str,df1.index.strftime('%Y-%m-%d')))
    
            y_high_stock1=list(df1['High'])
            y_open_stock1=list(df1['Open'])
            y_low_stock1=list(df1['Low'])
            y_close_stock1=list(df1['Close']) 
            y_volume_stock1=list(df1['Volume'])

            df2 = yf.download(stocks2, start_date, close_date)
            df2['Date']=df2.index.strftime('%d-%m-%Y')
    
            x_stock2=list(map(str,df2.index.strftime('%Y-%m-%d')))
        
            y_high_stock2=list(df2['High'])
            y_open_stock2=list(df2['Open'])
            y_low_stock2=list(df2['Low'])
            y_close_stock2=list(df2['Close'])  
            y_volume_stock2=list(df2['Volume'])
            x_final=x_stock2[:]
            if len(x_stock2)<len(x_stock1):
                y_high_stock2=y_high_stock2[-len(x_stock2):]
                y_open_stock2=y_open_stock2[-len(x_stock2):]
                y_low_stock2=y_low_stock2[-len(x_stock2):]
                y_close_stock2=y_close_stock2[-len(x_stock2):]
                y_volume_stock2=y_volume_stock2[-len(x_stock2):]
                x_final=x_stock2[:]
            elif len(x_stock2)>len(x_stock1) :
                y_high_stock1=y_high_stock1[-len(x_stock1):]
                y_open_stock1=y_open_stock1[-len(x_stock1):]
                y_low_stock1=y_low_stock1[-len(x_stock1):]
                y_close_stock1=y_close_stock1[-len(x_stock1):]
                y_volume_stock1=y_volume_stock1[-len(x_stock1):]
                x_final=x_stock1[:]
            context={
                'x':x_final,
                'y_high_stock1':y_high_stock1,
                'y_open_stock1':y_open_stock1,
                'y_low_stock1':y_low_stock1,
                'y_close_stock1':y_close_stock1,
                'y_high_stock2':y_high_stock2,
                'y_open_stock2':y_open_stock2,
                'y_low_stock2':y_low_stock2,
                'y_close_stock2':y_close_stock2,
                'y_volume_stock1':y_volume_stock1,
                'y_volume_stock2':y_volume_stock2,
                'company1':stocks1,
                'company2':stocks2,
                'df1':df1,
                'df2':df2,
                'max_price_stock1':round(max(y_high_stock1),2),
                'min_price_stock1':round(min(y_low_stock1),2),
                'last_day_price_stock1':round(y_close_stock1[-1],2),
                'change_in_price_stock1':round(y_high_stock1[-1]-y_high_stock1[0],2),
                'change_in_precentage_stock1':round(((y_high_stock1[-1]-y_high_stock1[0])/y_high_stock1[0])*100,2),
                'change_color1': 'red' if round(((y_high_stock1[-1]-y_high_stock1[0])/y_high_stock1[0])*100,2) < 0 else 'green',
                'max_price_stock2':round(max(y_high_stock2),2),
                'min_price_stock2':round(min(y_low_stock2),2),
                'last_day_price_stock2':round(y_close_stock2[-1],2),
                'change_in_price_stock2':round(y_high_stock2[-1]-y_high_stock2[0],2),
                'change_in_precentage_stock2':round(((y_high_stock2[-1]-y_high_stock2[0])/y_high_stock2[0])*100,2),
                'change_color2': 'red' if round(((y_high_stock2[-1]-y_high_stock2[0])/y_high_stock2[0])*100,2) < 0 else 'green',
                'flag':True,
                "start_date":start_date,
                "close_date":close_date
            }
        except Exception as e:
            context = {
                "flag": False,
                "err" : "Something went wrong while fetching data",
                "errSolution": "Please check internet connection or company code"
            }
    return render(request,'compare.html',context)

def predict(request):
    global df
    close_date= datetime.now()
    start_date= datetime(close_date.year-15, close_date.month, close_date.day)
    context={"flag":False}
    if request.method == 'POST':
        try:
            stocks = request.POST.get('company1')
            days=int(request.POST.get('days'))
            df = yf.download(stocks, start_date, close_date)

            x=list(map(str,df.index.strftime('%Y-%m-%d')))
            y_high=list(df['Close'])
            df=df.drop(['Open','High','Volume','Low','Adj Close'],axis=1)
            min_max_scalar=MinMaxScaler(feature_range=(0,1))
            data=df.values
            scaled_data=min_max_scalar.fit_transform(data)
            train_data=scaled_data[:,:]
            x_train=[]
            y_train=[]
            interval=90
            for i in range(interval,len(train_data)):
                x_train.append(train_data[i-interval:i,0])
                y_train.append(train_data[i,0])
            x_train,y_train=np.array(x_train),np.array(y_train)
            x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

            stop = EarlyStopping(
                monitor='val_loss', 
                mode='min',
                patience=5
            )
            checkpoint= ModelCheckpoint(
                filepath='./model_checkpoint.weights.h5',
                save_weights_only=True,
                monitor='val_loss',
                mode='min',
                save_best_only=True
            )
            model=Sequential()
            model.add(LSTM(200,return_sequences=True,input_shape=(x_train.shape[1],1)))
            model.add(LSTM(units=100))
            model.add(Dense(100))
            model.add(Dense(1))
            adam = optimizers.Adam(learning_rate=0.0005)
            model.compile(optimizer=adam, loss='mean_squared_error')
            model.fit(x_train, y_train, batch_size=512, epochs=10, shuffle=True, validation_split=0.05, callbacks = [checkpoint,stop])
            model.load_weights("./model_checkpoint.weights.h5")
            
            df_test = yf.download(stocks, start_date, close_date)
            df_test=df_test.drop(['Open','High','Volume','Low','Adj Close'],axis=1)
            predicted=[]
            for i in range(days):
                if predicted!=[]:
                    if (-interval+i)<0:
                        test_value=df_test[-interval+i:].values
                        test_value=np.append(test_value,predicted)
                    else:
                        test_value=np.array(predicted)
                else:
                    test_value=df_test[-interval+i:].values
                test_value=test_value[-interval:].reshape(-1,1)
                test_value=min_max_scalar.transform(test_value)
                test=[]
                test.append(test_value)
                test=np.array(test)
                test=np.reshape(test,(test.shape[0],test.shape[1],1))
                tomorrow_prediction=model.predict(test)
                tomorrow_prediction=min_max_scalar.inverse_transform(tomorrow_prediction)
                predicted.append(tomorrow_prediction[0][0])
            predicted_x=[]
            for i in range(1,days+1):
                predicted_x.append( str((date.today() + timedelta(days=i)).strftime('%Y-%m-%d')))
            if predicted[0]<predicted[-1] and y_high[-1]<predicted[-1]:
                buy="Yes ✅"
            else:
                buy="No ❌"
            dic={}
            dic['Date']=predicted_x
            dic['Prediction']=predicted
            df=pd.DataFrame.from_dict(dic)
            context={
                'x':x,
                'y_high':y_high,
                'company':stocks,
                'predicted_x':predicted_x,
                'predicted_y':predicted,
                "flag":True,
                "days":days,
                "csv":zip(predicted_x,predicted),
                "max_price":round(max(predicted),2),
                "min_price":round(min(predicted),2),
                "buy":buy,
                "change_in_precentage":round(((max(predicted)-min(predicted))/(min(predicted)))*100,2),
                "change_in_price":round((max(predicted)-min(predicted)),2),
                "change_color": 'green' if predicted[0]<predicted[-1] and y_high[-1]<predicted[-1] else 'red'
            }
        except Exception as e:
            context = {
                "flag": False,
                "err" : "Something went wrong while fetching data",
                "errSolution": "Please check internet connection or company code"
            }
    return render(request,'predict.html',context)
    
def download(request,id):
    global df,df1,df2
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="data.csv"'
    
    if id=='0': #download for details
        writer = csv.writer(response)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for ind in df.index:
            writer.writerow([ind,df['Open'][ind],df['High'][ind],df['Low'][ind],df['Close'][ind],df['Volume'][ind]])
    elif id=='1': #download for compare company-1
        writer = csv.writer(response)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for ind in df1.index:
            writer.writerow([ind,df1['Open'][ind],df1['High'][ind],df1['Low'][ind],df1['Close'][ind],df1['Volume'][ind]])
    elif id=='2': #download for compare company-2
        writer = csv.writer(response)
        writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        for ind in df2.index:
            writer.writerow([ind,df2['Open'][ind],df2['High'][ind],df2['Low'][ind],df2['Close'][ind],df2['Volume'][ind]])
    elif id=='3': #download for prediction
        writer = csv.writer(response)
        writer.writerow(['', 'Date','Prediction'])
        for ind in df.index:
            writer.writerow([ind,df['Date'][ind],df['Prediction'][ind]])
    return response
