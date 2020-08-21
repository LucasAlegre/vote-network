import http.client

conn = http.client.HTTPSConnection("dadosabertos.camara.leg.br")

headers = { 'user-agent': "vscode-restclient" }

conn.request("GET", "/api/v2/proposicoes?ordem=ASC&ordenarPor=id", headers=headers)

res = conn.getresponse()
data = res.readlines()

decoded = data

print(type(decoded))
print(decoded[0])