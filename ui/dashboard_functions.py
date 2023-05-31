import numpy as np
import pandas as pd

def simulate_poisson_arrivals(lam, n_samples=100):
    x = np.random.poisson(lam, n_samples)
    return x


def get_monthly_arrival_rates(path='data/Dataset casus GGZ - 2018_2019_2020.xls'):

  df = read_and_clean_data(path)

  df_statistics = DetermineArrivalRatePerMonth(df)

  dict_acuut,dict_semi_acuut, dict_regulier = getMonthlyArrivalRates(df)

  return {'acute' : dict_acuut
          , 'semiacute' : dict_semi_acuut
          , 'regular' : dict_regulier}




def read_and_clean_data(path):

  df = pd.read_excel(path, usecols = ['Patiënt', 'Nieuwe patiënt of niet', 'Type opname',
       'Herkomst nieuwe opname', 'Opname op type afdeling',
       'Opname op afdeling', 'Verantwoordelijke GGZ-instelling', 'Beweging',
       'Type beweging', 'Jaar start opname', 'Datum op wachtlijst',
       'Dag opname', 'Startdatum opname (of datum afvoeren van wachtlijst)',
       'Einddatum opname', 'Type ontslag', 'Doorplaatsing is naar',
       'Dag ontslag', 'Aantal opnamedagen afdeling',
       'Totale opnameduur spoedklinieken', 'Aantal keer overgeplaatst',
       'Heropname binnen 6 mnd',
       'Bij heropname aantal dagen sinds laatste ontslag', 'Soort opname',
       'Urgentie opname', 'Alleen op TOA', 'Opmerking'])
  df = df[df['Aantal opnamedagen afdeling'].apply(np.isreal)]
  #Rows contain unrelevant information for the current project.
  df = df[df['Type beweging']!='Geen opname'] 
  #Deze opnames zijn dus niet doorgegaan, bv omdat het niet meer nodig was of omdat de patiënt ergens anders is opgenomen, 
  #omdat het te lang duurde.
  df = df[(df["Opname op type afdeling"]!= 'nb') & (df['Urgentie opname']!='n.b.')]
  df = df[df['Jaar start opname']!=2017]

  df['Startdatum opname (of datum afvoeren van wachtlijst)'] = pd.to_datetime(df['Startdatum opname (of datum afvoeren van wachtlijst)'])
  df['month'] = df['Startdatum opname (of datum afvoeren van wachtlijst)'].map(lambda x: x.month)

  df['Startdatum opname (of datum afvoeren van wachtlijst)'] = pd.to_datetime(df['Startdatum opname (of datum afvoeren van wachtlijst)'])
  df['month'] = df['Startdatum opname (of datum afvoeren van wachtlijst)'].map(lambda x: x.month)

  return df



def DetermineArrivalRatePerMonth(df):
    df_statistics = pd.DataFrame()
    for year in df['Jaar start opname'].unique():
        for month in df['month'].unique():
            extract_info_df = df[(df['month']==month) & (df['Jaar start opname']==year)]
            df_statistics =  df_statistics.append({
                'Date': f'01-{month}-{int(year)}',
                'number of arrival': extract_info_df.shape[0]
                }, ignore_index= True)
    df_statistics['Date'] = pd.to_datetime(df_statistics['Date'], format='%d-%m-%Y')
    df_statistics['number of arrival'] = df_statistics['number of arrival'].astype(int) 
    df_statistics = df_statistics.sort_values(by='Date')
    df_statistics = df_statistics.set_index('Date')
    return df_statistics



def getMonthlyArrivalRates(df):
    """"" returns the seasonal arrival rates in a dictonary for acute, semi-acute and regulier patients. 
    """""
    
    #determining the latest admission date and filtering on the most recent year.
    df['Startdatum opname (of datum afvoeren van wachtlijst)'] = pd.to_datetime(df['Startdatum opname (of datum afvoeren van wachtlijst)'])
    latest_admission_date = df['Startdatum opname (of datum afvoeren van wachtlijst)'].max()
    start_admission_date = latest_admission_date -  pd.offsets.DateOffset(days=364)
    df= df[(df['Startdatum opname (of datum afvoeren van wachtlijst)'] >= start_admission_date ) & (df['Startdatum opname (of datum afvoeren van wachtlijst)'] <= latest_admission_date)]
    
    # separating the dataframes into three dataframes per patient type
    acuut_df = df[(df['Urgentie opname']=='Acuut')]
    semi_acuut_df = df[(df['Urgentie opname']=='Semi-acuut')]
    regulier_df=  df[(df['Urgentie opname']=='Regulier')]
    
    #Calculating the arrival rate per month per patient type
    arrival_rates_per_month_df_acuut  = DetermineArrivalRatePerMonth(acuut_df)
    arrival_rates_per_month_df_semi_acuut  = DetermineArrivalRatePerMonth(semi_acuut_df)
    arrival_rates_per_month_df_regulier  = DetermineArrivalRatePerMonth(regulier_df)
     
    #setting index to the month fully written    
    arrival_rates_per_month_df_acuut.index = arrival_rates_per_month_df_acuut.index.strftime('%B')
    arrival_rates_per_month_df_semi_acuut.index = arrival_rates_per_month_df_semi_acuut.index.strftime('%B')
    arrival_rates_per_month_df_regulier.index = arrival_rates_per_month_df_regulier.index.strftime('%B')
    
    # creating the dictonaries with the months 
    dict_acuut = arrival_rates_per_month_df_acuut['number of arrival'].to_dict()
    dict_semi_acuut = arrival_rates_per_month_df_semi_acuut['number of arrival'].to_dict()
    dict_regulier  =  arrival_rates_per_month_df_regulier['number of arrival'].to_dict()  
    
    return dict_acuut,dict_semi_acuut, dict_regulier