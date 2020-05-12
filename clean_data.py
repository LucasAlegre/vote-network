#%%
import pandas as pd

# Mandato inteiro
start_date_filter = '01-02-2015'
end_date_filter = '31-01-2019'

start_year = 2015
years = [x + start_year for x in range(0, 6)]
all_data = None

for year in years:
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

#%% PMDB changed name to MDB
all_data['deputado_siglaPartido'].replace('PMDB', 'MDB', inplace=True)

#%% 
all_data['deputado_siglaPartido'].fillna('Sem Partido', inplace=True)

#%%
# all_data.groupby('idVotacao')['voto'].count()

#%%
all_data.to_csv('resources/votos_{}_to_{}.csv'.format(start_date_filter, end_date_filter), index=False)
