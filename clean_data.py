#%%
import pandas as pd
import numpy as np
import urllib.request
import os
import argparse

YEAR = 2

parser = argparse.ArgumentParser(description="Gets data from the API")
parser.add_argument('-s', "--start", type=str,
                action="store", dest="start_date", default="31-01-2019",
                help="DD-MM-YYYY  Start data for the period that you want to collect the data.")
parser.add_argument('-e', "--end", type=str,
                action="store", dest="end_date", default="30-12-2020",
                help="DD-MM-YYYY  End data for the period that you want to collect the data.")

args = parser.parse_args()

start_date, end_date = args.start_date, args.end_date


# Mandato inteiro
start_date_filter: str = start_date#'31-01-2019' # '01-02-2015'
end_date_filter: str = end_date #'30-12-2020'

start_year = int(start_date_filter.split('-')[YEAR])
end_year = int(end_date_filter.split('-')[YEAR])
years = [x + start_year for x in range(end_year - start_year + 1)]
all_data = None

for year in years:
    urllib.request.urlretrieve('https://dadosabertos.camara.leg.br/arquivos/votacoesVotos/csv/votacoesVotos-{}.csv'.format(year), 'resources/votacoesVotos-{}.csv'.format(year))
    os.system('gzip -f resources/votacoesVotos-{}.csv'.format(year))

    data = pd.read_csv('resources/votacoesVotos-' + str(year) + '.csv.gz', compression='gzip', sep=';')

    del data['uriVotacao']
    del data['deputado_urlFoto']
    del data['deputado_uri']
    del data['deputado_uriPartido']
    del data['deputado_idLegislatura']

    data = data[data['voto'] != 'Simbólico']

    data['dataHoraVoto'] = pd.to_datetime(data['dataHoraVoto'])    

    mask = (data['dataHoraVoto'] >= start_date_filter) & (data['dataHoraVoto'] <= end_date_filter)
    data = data.loc[mask]
    data = data.sort_values(by=['dataHoraVoto'])
    
    if year == start_year:
        all_data = data
    else:
        all_data = pd.concat([all_data, data])

#%% Take care of different names for same deputy
for group, df_group in all_data.groupby('deputado_id'):
    all_data['deputado_nome'].loc[all_data['deputado_id'] == group] = sorted(df_group['deputado_nome'].unique())[0]

# Esse cara tá sem partido por algum motivo, mas no google ele é do solidariedade
all_data['deputado_siglaPartido'].loc[all_data['deputado_nome'] == 'Simplício Araújo'] = 'SOLIDARIEDADE'

#%% Partidos que mudaram de nome
all_data['deputado_siglaPartido'].replace('PMDB', 'MDB', inplace=True)
all_data['deputado_siglaPartido'].replace('PRB', 'REPUBLICANOS', inplace=True)
all_data['deputado_siglaPartido'].replace('PR', 'PL', inplace=True)
all_data['deputado_siglaPartido'].replace('PATRIOTA', 'PATRI', inplace=True)
all_data['deputado_siglaPartido'].replace('PPS', 'CIDADANIA', inplace=True)
all_data['deputado_siglaPartido'].replace('PPL', np.nan, inplace=True) # PPL for incorporado
all_data['deputado_siglaPartido'].replace('PRP', np.nan, inplace=True) # PRP for incorporado
all_data['deputado_siglaPartido'].replace('PHS', np.nan, inplace=True) # PHS for incorporado

#%% 
#all_data['deputado_siglaPartido'].fillna('Sem Partido', inplace=True)

#%%
# all_data.groupby('idVotacao')['voto'].count()

#%%
all_data.to_csv('resources/votos_{}_to_{}.csv'.format(start_date_filter, end_date_filter), index=False)
