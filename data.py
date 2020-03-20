import requests
import pandas as pd


def get_votations(begin_date='2020-01-01', max_pages=100):
    """Get all votations data
    
    Args:
        begin_date (str, optional): Earliest date to look for votations 
        max_pages (int, optional): Max number of pages of 200 itens. Defaults to 100.
    Return:
        (dict, list): Dictionary with votation data and list of votations ids
    """    
    votations = []
    for page in range(1, max_pages+1):
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

    """ votations, votations_ids = get_votations()
    votes = {}
    for id in votations_ids:
        votes[id] = get_votes(id)  # some votations doesn't have votes information

    deputies = {}
    for vote_id in votes.keys():
        for vote in votes[vote_id]:
            if vote['deputado_']['id'] not in deputies:
                deputies[vote['deputado_']['id']] = vote['deputado_']

    print(deputies, len(deputies))  # Ta printando 518 deputados, tem 513 na camara! (ué) """

    df = pd.read_csv('votacoesVotos-2020.csv', sep=';')
    print(len(df['deputado_id'].unique()))  # Aqui dá 508 deputados, os csv não bate com o resultado da API REST



