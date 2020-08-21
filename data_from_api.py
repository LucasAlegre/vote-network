import requests
import pandas as pd
import argparse
from progress.bar import Bar



TEMAS = {
    "educacao": "46",
    "saude": "56",
    "economia": "40"
}

URI = "https://dadosabertos.camara.leg.br"

def get_proposicoes(begin_date='2020-01-01', end_date=None, max_pages=300, theme="saude"):

    bar = Bar('Getting propositions', max=max_pages)

    proprosicoes = []
    page_count = 0
    for page in range(1, max_pages+1):
        page_count += 1
        if end_date is not None:
            response = requests.get('{}/api/v2/proposicoes?dataInicio={}&dataFim={}&pagina={}&itens=100&ordem=ASC&ordenarPor=id&codTema={}'.format(URI, begin_date, end_date, page,TEMAS[theme]))
            
        else:
            response = requests.get('{}/api/v2/proposicoes?dataInicio={}&pagina={}&itens=100&ordem=ASC&ordenarPor=id&codTema={}'.format(URI, begin_date, page, TEMAS[theme]))
      
        data = response.json()['dados']
        if data:
            proprosicoes.extend(data)
        else:
            break
        bar.next()
    bar.finish()
    proprosicoes_ids = [p['id'] for p in proprosicoes]


    unique_list = list(set(proprosicoes_ids))

    print(unique_list)

    with open("resources/list_od_proposicoes_{}_{}.txt".format(theme, page_count), 'w') as file:
        for item in unique_list:
            file.write("%s\n" % item)

    return proprosicoes, unique_list



def get_votacoes_ids_from_proposicoes(proposicoes_ids: list = []):

   

    resource: str = "/api/v2/proposicoes/{}/votacoes?ordem=ASC"
  
    votacoes = []
    for prop in proposicoes_ids:
        print(URI + resource.format(str(prop)))
        response = requests.get(URI + resource.format(str(prop)))
        data = response.json()['dados']

        if data:
            votacoes.extend(data)

    votacoes_ids = [v['id'] for v in votacoes]

    


    return votacoes_ids



def get_votations(begin_date='2020-01-01', end_date=None, max_pages=100):
    """Get all votations data
    
    Args:
        begin_date (str, optional): Earliest date to look for votations 
        max_pages (int, optional): Max number of pages of 200 itens. Defaults to 100.
    Return:
        (dict, list): Dictionary with votation data and list of votations ids
    """    
    votations = []
    for page in range(1, max_pages+1):
        if end_date is not None:
            response = requests.get('https://dadosabertos.camara.leg.br/api/v2/votacoes?dataInicio={}&dataFim={}&pagina={}&itens=200&ordem=DESC&ordenarPor=dataHoraRegistro'.format(begin_date, end_date, page))
        else:
            response = requests.get('https://dadosabertos.camara.leg.br/api/v2/votacoes?dataInicio={}&pagina={}&itens=200&ordem=DESC&ordenarPor=dataHoraRegistro'.format(begin_date, page))

        data = response.json()['dados']
        if data:
            votations.extend(data)
        else:
            break

    votations_ids = [v['id'] for v in votations]
    return votations, votations_ids

def get_votation_details(votation_id):
    """Return detailed description of votation
    
    Args:
        votation_id (str): The votation id
    Returns:
        dict: Detailed description
    """    
    response = requests.get('https://dadosabertos.camara.leg.br/api/v2/votacoes/{}'.format(votation_id))
    return response.json()['dados']

def get_votes(votation_id):
    """Get votes
    
    Args:
        votation_id (str): Votation id
    
    Returns:
        (dict): The votes
    """    
    response = requests.get('https://dadosabertos.camara.leg.br/api/v2/votacoes/{}/votos'.format(votation_id))
    return response.json()['dados']


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Gets data from the API")
    parser.add_argument('-s', "--start", type=str,
                        action="store", dest="start_date", default="2020-01-01",
                        help="YYYY-MM-DD  Start data for the period that you want to collect the data.")
    parser.add_argument('-e', "--end", type=str,
                        action="store", dest="end_date", default=None,
                        help="YYYY-MM-DD  End data for the period that you want to collect the data.")


    parser.add_argument('-t', "--theme", type=str,
                        action="store", dest="proposition_themes", default=None,
                        help="Desired theme for the proposition collect (saude/educacao/economia). Default: None")

    args = parser.parse_args()

    start_date, end_date, theme = args.start_date, args.end_date, args.proposition_themes

    if (theme is None):
        votations, votations_ids = get_votations(begin_date = start_date, end_date = end_date)
        votes = {}
        for id in votations_ids:
            votes[id] = get_votes(id)  # some votations do not have votes information

        deputies = {}
        for vote_id in votes.keys():
            for vote in votes[vote_id]:
                if vote['deputado_']['id'] not in deputies:
                    deputies[vote['deputado_']['id']] = vote['deputado_']

        print(deputies, len(deputies))  # Ta printando 518 deputados, tem 513 na camara! (ué) """

        df = pd.read_csv('votacoesVotos-from_{}-to-_{}.csv'.format(start_date, end_date), sep=';')
        print(len(df['deputado_id'].unique()))  # Aqui dá 508 deputados, os csv não bate com o resultado da API REST
    else:
        proposicoes, proposicoes_ids = get_proposicoes(begin_date=start_date, end_date=end_date, theme=theme)
        
        import multiprocessing as mp

        pool = mp.Pool(mp.cpu_count())

        votacoes_ids = pool.map(get_votacoes_ids_from_proposicoes, [[proposition] for proposition in proposicoes_ids])
        
        votacoes_ids = [item for sublist in votacoes_ids for item in sublist]
        votacoes_ids = list(set(votacoes_ids))
        print(votacoes_ids)
        print(len(votacoes_ids))

        with open("resources/votacoes_{}_{}_to_{}.txt".format(theme, start_date, end_date), 'w') as file:
            for item in votacoes_ids:
                file.write("%s\n" % item)

