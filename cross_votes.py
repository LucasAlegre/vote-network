
import pandas as pd

data = pd.read_csv('resources/all.csv')

deputados = data["deputado_nome"]
votacoes = pd.unique(data["idVotacao"])

affinity = {}


print(len(votacoes) * len(deputados) + len(deputados))

# for votacao in votacoes:
#     for deputado_1 in deputados:
#         for deputado_2 in deputados:
#                 affinity[(deputado_1, deputado_2)] = 0


# for votacao in votacoes:
#     for deputado_1 in deputados:
#         filter = data["idVotacao"] == votacao
#         voto_deputado_1 = data.where(filter)
#         for deputado_2 in deputados:
#             if deputado_1 != deputado_2:
#                 affinity[(deputado_1, deputado_2)] += 1
