import pandas as pd
from util import get_votations_theme
import random
import numpy as np


    

path = 'resources/votos_31-01-2019_to_30-12-2020.csv'
df = pd.read_csv(path)



votacoes = get_votations_theme("saude", "2020-01-01", "2020-07-20")
votacoes.extend(get_votations_theme("educacao", "2020-01-01", "2020-07-20"))
votacoes.extend(get_votations_theme("economia", "2020-01-01", "2020-07-20"))

votacoes.extend(get_votations_theme("educacao", "2019-01-01", "2019-07-20"))
votacoes.extend(get_votations_theme("economia", "2019-01-01", "2019-07-20"))
votacoes.extend(get_votations_theme("educacao", "2019-01-01", "2019-07-20"))


saude_2020 = get_votations_theme("saude", "2020-01-01", "2020-07-20")
educ_2020 = (get_votations_theme("educacao", "2020-01-01", "2020-07-20"))
econo_2020 = (get_votations_theme("economia", "2020-01-01", "2020-07-20"))

saude_2019=(get_votations_theme("educacao", "2019-01-01", "2019-07-20"))
educ_2019=(get_votations_theme("economia", "2019-01-01", "2019-07-20"))
econo_2019=(get_votations_theme("educacao", "2019-01-01", "2019-07-20"))



print(len(list(set(votacoes))))

votacoes = list(set(votacoes))





are_not_in = []
are_in = []

votes_that_we_have = df["idVotacao"].unique().tolist()

print()
print("Set sizes")
print("Educ 2019:", len(set(educ_2019) & set(votes_that_we_have)))
print("Saude 2019:", len(set(saude_2019) & set(votes_that_we_have)))
print("Economia 2019:", len(set(econo_2019) & set(votes_that_we_have)))

print("Educ 2020:", len(set(educ_2020) & set(votes_that_we_have)))
print("Saude 2020:", len(set(saude_2020) & set(votes_that_we_have)))
print("Econo 2020:", len(set(econo_2020) & set(votes_that_we_have)))
print("-----------")

number_of_elements_to_sort = 12

educ_2019 = set(educ_2019) & set(votes_that_we_have)
saude_2019 = set(saude_2019) & set(votes_that_we_have)
econo_2019 = set(econo_2019) & set(votes_that_we_have)

educ_2020 = set(educ_2020) & set(votes_that_we_have)
saude_2020 = set(saude_2020) & set(votes_that_we_have)
econo_2020 = set(econo_2020) & set(votes_that_we_have)


educ_2019_sample = random.sample(set(educ_2019), 12)
saude_2019_sample = random.sample(set(saude_2019), 12)
econo_2019_sample = random.sample(set(econo_2019), 12)

educ_2020_sample = random.sample(set(educ_2020), 12)
saude_2020_sample = random.sample(set(saude_2020), 12)
econo_2020_sample = random.sample(set(econo_2020), 12)


np.save("resources/samples/educacao_2019_sample" , educ_2019_sample)
np.save("resources/samples/saude_2019_sample" , saude_2019_sample)
np.save("resources/samples/economia_2019_sample" , econo_2019_sample)

np.save("resources/samples/educacao_2020_sample" , educ_2020_sample)
np.save("resources/samples/saude_2020_sample" , saude_2020_sample)
np.save("resources/samples/economia_2020_sample" , econo_2020_sample)


print(educ_2019_sample)


print(len((set(educ_2019) - set(econo_2020)) & set(votes_that_we_have)))

for votacao in votacoes:
    if votacao not in votes_that_we_have:
        are_not_in.append(votacao)
    else:
        are_in.append(votacao)

print((len(are_in), len(are_not_in)))