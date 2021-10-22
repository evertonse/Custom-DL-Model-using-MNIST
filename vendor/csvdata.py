import re
import csv as c
import utils






# ##-Funções Core-##
def data(url, sanitizefn=utils.sanitizecsv, filter=lambda x: tuple(x), delim=","):

    # Pega as linhas de um arquivo na ordem de leitura
    def getLinhas(url):
        linhas = []
        labels = []
        with open(url) as file:
            lines = c.reader(file, delimiter=delim)
            # Loop de Leitura de linha
            for i,line in enumerate(lines):
                if i == 0:
                    labels = line
                    continue
                # Aplicamos umas limpeza na linha, função passada como argumento porém não aplicamos caso seja a primeira linha;
                # A lista é filtrada, e retorna uma lista com items desejados
                if filter(line):
                    # filter deve retornar oq items que deseja, caso nenhum é desejado e retorne false or [], então não será adicionado a linhas
                    linhas.append(filter(sanitizefn(line)))
        # Retornamos a tupla dessas linhas
        return (tuple(linhas),labels)


    ### Private Fields ###
    # lista de linhas a seresm usadas nas outras funções
    linhas,labels = getLinhas(url) 
    ### ############## ###


    
    # Pega apenas o labels, ou seja a primeira linha do csv e retornas numas lista
    def getLabels():
        linha = linhas  # External Data Becareful
        lista = []
        for i in linha[0]:
            lista.append(i)
        *labels, last = lista
        return [*labels, last]


    # Pega dados unicos de uma coluna EX: todos os esportes que existe na coluna "Sports"
    def getUnique(coluna: int):
        linha = linhas  # External Data Becareful
        uniques = set()
        for i in linha:
            uniques.add(i[coluna])
        return tuple(uniques)

    ##### NÃO TESTADO AINDA NÃO USAR"
    # Pega colunas a partir de indices que indicam quantas colunas quer ter
    def getColunas(*colunas):
        dados = linhas  # External Data Becareful
        lista = []
        column = {}
        for i in colunas:
            for item in dados:
                lista.append(item[i])
            column[i] = lista
            lista = []
        return column

    def getAxis(filter=lambda x: x, tipo="Q", finalcb=lambda x: x,sort=True):
        dados = linhas  # external data, becareful
        # tipo pode ser "Q", "M", "T" -> T=total, Q=quantidade, M=media   (case insensitive)(pegamos apenas a primeira letra sem espaços do "tipo")
        tipo = tipo.strip().upper()[0]
        qnt, total = [{}, {}]
        numeric = False
        if tipo == "M" or tipo == "T":
            numeric = True
        for item in dados[1:]:
            # axis é o esperado uma tuple de 2 valores, cada valor é um dos axis
            axis = filter(item)
            # caso false ou vazio seja retornado do axis, então não será contado
            if(axis):
                # Se for numerico essa axis, façamos um total dele
                if numeric:
                    # Se axis[1] for "Na" ou qualquer coisa que não seja numerica pulamos a contagem
                    # OBS note que apenas pulamos contagem se for esperado que seja numerico
                    # Se for o caso de for uma palavra, simplesmente adicionamos mais um a {qnt} em baixo
                    if not str(axis[1]).isnumeric():
                        print("I'm  skipped", axis)
                        continue
                    adition = int(axis[1])
                    total[axis[0]] = total.get(axis[0], 0) + adition
                # contamos a quantidade de ocorrencia de de certo axis1 nem axis0
                qnt[axis[0]] = qnt.get(axis[0], 0)+1
        # handling returns
        if tipo == "Q":
            return utils.sortdict(finalcb(qnt),sort=sort)

        if numeric:
            if tipo == "M":
                return utils.sortdict(finalcb({key: value/qnt[key] for key, value in total.items()}),sort=sort)
            elif tipo == "T":
                return utils.sortdict(finalcb(total),sort=sort)
            else:
                return utils.sortdict(finalcb(qnt),sort=sort)
            # total divido por quantos tem


    ### Dados para passar para o dictionario csv
    csv = {
        'linhas': linhas,  # retorna uma lista de  cada linha:list
        # retorna um dicionario de colunas com valor de lista de  cada coluna:list
        'colunas': getColunas,
        'labels': getLabels,  # retorna uma lista de  cada label:str
        'axis':  getAxis,  # {label:[]}
        'unique':  getUnique,  # Pega dados unicos de umas coluna
    }
    return csv
