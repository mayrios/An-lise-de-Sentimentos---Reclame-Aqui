from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
import pickle


def formatar_sentenca(sentenca):
   return {palavra: True for palavra in word_tokenize(sentenca)}

f_insast = open('insastifeito.txt', 'rb')
insatisfeitos = f_insast.read().splitlines()
f_insast.close()

f_indig = open('indignado.txt', 'rb')
indignados = f_indig.read().splitlines()
f_indig.close()

dados_treinamento = []

for instafisfacao in insatisfeitos:
   dados_treinamento.append([formatar_sentenca(instafisfacao.decode("utf8").lower()), "insatisfeito"])

for indignacao in indignados:
   dados_treinamento.append([formatar_sentenca(indignacao.decode("utf8").lower()), "indignado"])
   
modelo = NaiveBayesClassifier.train(dados_treinamento)	

with open('modelo.obj', 'wb') as f:
    modelo_serial = pickle.dump(modelo, f)
    print('Modelo classificador treinado e armazenado em modelo.obj')
#nltk.download()


