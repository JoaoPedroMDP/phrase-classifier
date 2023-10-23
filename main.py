#  coding: utf-8
import argparse
import csv
import pickle
from datetime import datetime

import pandas
import json

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from config import MODEL_FILENAME, SPLIT_INTO_TRAIN_TEST


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

    if SPLIT_INTO_TRAIN_TEST:
        train = frases_formatadas[:int(len(frases_formatadas) * 0.8)]
        test = frases_formatadas[int(len(frases_formatadas) * 0.8):]
        df = pandas.DataFrame(train)
        df.to_csv("dados_formatados/frases_vetorizadas_train.csv", index=False, header=False)

        df = pandas.DataFrame(test)
        df.to_csv("dados_formatados/frases_vetorizadas_test.csv", index=False, header=False)


def formata_classif():
    classif_das_frases = open("DADOS/CLtx.dat", "r").readlines()
    classif_formatadas = []
    for classif in classif_das_frases:
        classif_formatadas.append(int(float(classif)))

    if SPLIT_INTO_TRAIN_TEST:
        train = classif_formatadas[:int(len(classif_formatadas) * 0.8)]
        test = classif_formatadas[int(len(classif_formatadas) * 0.8):]
        df = pandas.DataFrame(train)
        df.to_csv("dados_formatados/classificacao_das_frases_train.csv", index=False, header=False)

        df = pandas.DataFrame(test)
        df.to_csv("dados_formatados/classificacao_das_frases_test.csv", index=False, header=False)


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
    df.to_csv(f"dados_formatados/vetores_das_palavras.csv", index=False, header=False)


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


def sum_and_average_vectors(vect1: list, vect2: list):
    final_vector = [None] * len(vect1)
    for i in range(len(vect1)):
        final_vector[i] = (vect1[i] + vect2[i]) / 2
    
    return final_vector


def vectorize_phrase(frase: str):
    frase = frase.upper()
    print(frase)
    dicionario: dict = json.loads(open("dados_formatados/dicionario.json").read())
    words = frase.split(" ")
    final_vector = []

    for word in words:
        vector = dicionario.get(word, None)
        if not vector:
            print(f"Não achou {word}")
            continue

        if not final_vector:
            final_vector = vector

        final_vector = sum_and_average_vectors(final_vector, vector)

    return final_vector


def get_data(mode: str):
    with open(f"dados_formatados/frases_vetorizadas_{mode}.csv", "r") as f:
        rows = list(csv.reader(f))
        vectorized_phrases = []
        for row in rows:
            vectorized_phrases.append([float(x) for x in row])
    
    with open(f"dados_formatados/classificacao_das_frases_{mode}.csv", "r") as f:
        phrases_classes = [int(x[0]) for x in list(csv.reader(f))]
    
    return vectorized_phrases, phrases_classes


def look_for_trained_models():
    filename = MODEL_FILENAME
    try:
        loaded_model = pickle.load(open(filename, "rb"))
        print("Modelo carregado!!")
        return loaded_model
    except FileNotFoundError:
        print("Não existem modelos prontos.")
        return None


def apply_pca(data):
    pca = PCA(n_components=300)
    return pca.fit_transform(data)
    # return data


def get_model(test: bool = False):
    model = look_for_trained_models() if not test else None
    if not model or test:
        print("Não existe modelo pronto, treinando...")
        print("Pegando dados de teste")
        vectorized_phrases, phrases_classes = get_data("train")

        reduced = apply_pca(vectorized_phrases)

        print("Treinando modelo")
        time = datetime.now()
        # model: SVC = SVC().fit(reduced, phrases_classes)
        # model: GaussianNB = GaussianNB().fit(reduced, phrases_classes)
        # model: LogisticRegression = LogisticRegression().fit(reduced, phrases_classes)
        # model: RandomForestClassifier = RandomForestClassifier().fit(reduced, phrases_classes)
        # model: Perceptron = Perceptron().fit(reduced, phrases_classes)
        model: MLPClassifier = MLPClassifier((3, 5), max_iter=300).fit(reduced, phrases_classes)
        # model: MLPClassifier = MLPClassifier((3, 5, 6)).fit(reduced, phrases_classes)

        print(f"Tempo de treinamento: {datetime.now() - time}")
        print("Salvando modelo...")
        pickle.dump(model, open(MODEL_FILENAME, "wb"))

    return model


def test_model(model):
    print("Pegando dados de teste")
    vectorized_phrases, phrases_classes = get_data("test")
    reduced = apply_pca(vectorized_phrases)
    print("Testando modelo")
    print(model.score(reduced, phrases_classes))


def main(phrase: str, test: bool = False):
    # Primeiro eu formatei os arquivos pq misericórdia, .dat é de matar
    # os.makedirs("dados_formatados", exist_ok=True)

    # formata_frases()
    # formata_classif()
    # formata_palavras()
    # formata_vetores_palavras()

    # Agora quero construir um dicionario com as palavras e seus vetores
    # constroi_dicionario()

    # Farei um teste rápido pra ver se a indexação tá certinha
    # teste()

    # Vetorizador de frases
    # vector = vectorize_phrase("IMENSA IMENSO")

    # Retorna o modelo
    model = get_model(test)

    if test:
        print("Testando modelo")
        test_model(model)
        return

    vectorized_phrase = vectorize_phrase(phrase)
    reduced = apply_pca([vectorized_phrase])
    if len(vectorized_phrase) == 0:
        print(f"Não foi possível analisar a frase \"{phrase}\"")
        return

    result = model.predict([vectorized_phrase])
    print("Boa" if result == 1 else "Ruim")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='PhraseBadnessDetector',
                    description='Detecta se uma frase é boa ou ruim'
                    )
    parser.add_argument('--phrase', required=False, help='Frase a ser analisada.')
    parser.add_argument('--test', required=False, help='Se o modelo deve ser testado', default=False, action='store_true')
    args = parser.parse_args()
    main(args.phrase, args.test)
