import pandas as pd

data = pd.read_csv('resources/votacoesVotos-2020.csv.gz', compression='gzip', sep=';')

start_date_filter = '01-02-2015'
end_date_filter = '31-01-2019'

start_year = 2014
years = [x + start_year for x in range(0,6)]


for year in years:
    data = pd.read_csv('resources/votacoesVotos-' + str(year) + '.csv.gz', compression='gzip', sep=';')

    del data['uriVotacao']
    del data['deputado_urlFoto']
    del data['deputado_uri']
    del data['deputado_uriPartido']
    del data['deputado_id']
    del data['deputado_idLegislatura']

    data = data[data['voto'] != 'Simbólico']

    data['dataHoraVoto'] = pd.to_datetime(data['dataHoraVoto'])    

    mask = (data['dataHoraVoto'] >= start_date_filter) & (data['dataHoraVoto'] <= end_date_filter)
    data = data.loc[mask]
    data = data.sort_values(by=['dataHoraVoto'])
    
    if year == 2014:
        data.to_csv('all.csv')
    else:
        data.to_csv('all.csv', mode='a', header=False)