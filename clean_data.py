#%%
import pandas as pd
import numpy as np
import urllib.request
import os
import argparse

import pickle


YEAR = 2

def save_motion_themes(content: dict):
    with open('resources/motions_themes.json', 'w') as pfile:
        pfile.write(content.__str__())

def get_motions_propositions(years):
    
    all_data = None
    for year in years:
        
        '''
        Nestes arquivos, separados por ano de realização das votações, são listados os identificadores básicos de cada votação e de cada Proposição afetada por ela. 
        É normal uma votação gerar efeitos em mais de uma proposição. 
        Quando existente, é fornecido um campo com um texto descritivo do efeito gerado pela votação sobre a proposição.
        ''' 
        url = "http://dadosabertos.camara.leg.br/arquivos/votacoesProposicoes/{0}/votacoesProposicoes-{1}.{0}".format("csv", str(year))
        file = 'resources/votacoesProposicoes-{}.csv'.format(year)
        
        if not os.path.isfile(file + ".gz"):
            urllib.request.urlretrieve(url, file)
            os.system("gzip -f {}".format(file))

        data = pd.read_csv(file + '.gz', compression='gzip', sep=';')
        if year == years[0]:
            all_data = data
        else:
            all_data = pd.concat([all_data, data])
        
    return all_data.rename(columns={"proposicao_uri": "uriProposicao"})

def get_themes(props: pd.DataFrame):

    props = props[props["proposicao_ano"].notnull()]

    years = props["proposicao_ano"].unique()
    new = []
    for element in years:
        new.append(int(element))
    

    years = new

    years = sorted(years)
    for year in years:
        url = "http://dadosabertos.camara.leg.br/arquivos/proposicoesTemas/{0}/proposicoesTemas-{1}.{0}".format("csv", year)

        all_data = None

        file = 'resources/proposicoesTemas-{}.csv'.format(year)
        if not os.path.isfile(file + ".gz"):
            urllib.request.urlretrieve(url, file)
            os.system('gzip -f {}'.format(file))
        data = pd.read_csv(file + '.gz', compression='gzip', sep=';')

        


        if year == years[0]:
            all_data = data
        else:
            all_data = pd.concat([all_data, data])
        
    return all_data


def get_motions_data(year):

    file = 'resources/votacoesVotos-{}.csv'.format(year)

    if not os.path.isfile(file + ".gz"):
        urllib.request.urlretrieve('https://dadosabertos.camara.leg.br/arquivos/votacoesVotos/csv/votacoesVotos-{}.csv'.format(year), 'resources/votacoesVotos-{}.csv'.format(year))
        os.system('gzip -f {}'.format(file))
    return pd.read_csv('{}.gz'.format(file), compression='gzip', sep=';')




def merge_motion_theme(motion_prop: pd.DataFrame, prop_theme: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(motion_prop, prop_theme, on="uriProposicao", how="inner")   

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


motions_propositions = get_motions_propositions(years)
prop_themes = get_themes(motions_propositions)

motions_propositions.to_csv('resources/motions_propositions_{}_to_{}.csv'.format(start_date_filter, end_date_filter), index=False)
prop_themes.to_csv('resources/prop_themes_{}_to_{}.csv'.format(start_date_filter, end_date_filter), index=False)

motions_themes = merge_motion_theme(motions_propositions, prop_themes)[["idVotacao", "tema"]]


motion_to_themes = {}

for motionId, grouped in motions_themes.groupby(["idVotacao"]):
    motion_to_themes[motionId] = grouped["tema"].unique()

print(type(motion_to_themes))

save_motion_themes(motion_to_themes)

for year in years:

    data = get_motions_data(year)


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


all_data["theme"] = all_data["idVotacao"].map(motion_to_themes)

#%% 
#all_data['deputado_siglaPartido'].fillna('Sem Partido', inplace=True)

#%%
# all_data.groupby('idVotacao')['voto'].count()

#%%
all_data.to_csv('resources/votos_{}_to_{}.csv'.format(start_date_filter, end_date_filter), index=False)
