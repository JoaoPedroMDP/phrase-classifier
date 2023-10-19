#  coding: utf-8
import datetime
import os

import pandas
import json


def pega_vetores():
    return pandas.read_csv("dados_formatados/vetores_das_palavras.csv", header=None).itertuples(index=False)


def formata_frases():
    # Aqui vou pegar o .dat e converter em .csv
    frases = open("DADOS/WTEXT.dat", "r").readlines()
    frases_formatadas = []
    for frase in frases:
        splitado = frase.split(" ")
        linha = []
        for item in splitado:
            if item == '':
                continue
            linha.append(float(item))
        frases_formatadas.append(linha)

    df = pandas.DataFrame(frases_formatadas)
    df.to_csv("dados_formatados/frases_vetorizadas.csv", index=False, header=False)


def formata_classif():
    classif_das_frases = open("DADOS/CLtx.dat", "r").readlines()
    classif_formatadas = []
    for classif in classif_das_frases:
        classif_formatadas.append(int(float(classif)))

    df = pandas.DataFrame(classif_formatadas)
    df.to_csv("dados_formatados/classificacao_das_frases.csv", index=False, header=False)


def formata_palavras():
    palavras = open("DADOS/PALAVRAS.txt", "r", encoding="Latin-1").readlines()
    palavras_formatadas = []
    for palavra in palavras:
        palavras_formatadas.append(palavra.replace("\n", "").strip())

    with open("dados_formatados/palavras.txt", "w") as f:
        f.writelines("%s\n" % palavra for palavra in palavras_formatadas)


def formata_vetores_palavras():
    vetores_das_palavras = open("DADOS/WVECTS.dat", "r").readlines()
    vetores_formatados = []

    for vetor in vetores_das_palavras:
        splitado = vetor.split(" ")
        linha = []
        for item in splitado:
            if item == '':
                continue
            linha.append(float(item))
        vetores_formatados.append(linha)

    df = pandas.DataFrame(vetores_formatados)
    df.to_csv(f"dados_formatados/vetores_das_palavras{datetime.datetime.now()}.csv", index=False, header=False)


def constroi_dicionario():
    palavras = open("dados_formatados/palavras.txt").readlines()
    vetores = list(pega_vetores())
    dicionario = {}

    for i in range(len(vetores)):
        dicionario[palavras[i].strip()] = vetores[i]

    with open("dados_formatados/dicionario.json", "w") as f:
        json.dump(dicionario, f)


def teste():
    # Peguei a primeira e ultima palavra, e anotei o valor da posição 0 e -1 do vetor, e vou checar se bate
    palavras = {
        "IMENSA": [0.014095523, 0.016638779],
        "IMENSO": [-0.010602339, 0.018839753]
    }
    dicionario = json.load(open("dados_formatados/dicionario.json"))
    for palavra, valores in palavras.items():
        start = dicionario[palavra][0]
        end = dicionario[palavra][-1]
        if start == valores[0] and end == valores[1]:
            print(f"Palavra {palavra} está indexada corretamente")


def main():

    # Primeiro eu formatei os arquivos pq misericórdia, .dat é de matar
    os.makedirs("dados_formatados", exist_ok=True)

    formata_frases()
    formata_classif()
    formata_palavras()
    formata_vetores_palavras()

    # Agora quero construir um dicionario com as palavras e seus vetores
    # constroi_dicionario()

    # Farei um teste rápido pra ver se a indexação tá certinha
    # teste()


if __name__ == '__main__':
    main()
