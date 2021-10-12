

# Conta quantas vezes cada elemento de uma lista aparece em outra lista, 
def match(matches=[], lista=[]):
    count = 0
    for item in lista:
        for match in matches:
            if item == match:
                count += 1
    return count


# A partir de um dicionario, retorna uma lista de chaves e valores, em ordem
def sortdict(dic: dict, sort=True):
    if dic == None:
        print("Dictionary Vazio, Esqueceu de retornar algum dos callbacks?")
    if sort:
        items = sorted(dic.items())
    else:
        items = dic.items()
    # Sort dictionary intro a list of key and items
    dic = {n[0]: n[1] for n in items}
    return [dic.keys(), dic.values()]


# transforma uma lista-matriz de duas colunas de valores em um dicionário
# Ou seja, uma lista de keys e values de volta pra dictionary
def todict(lista:list,invert=False):
    chaves,valores = [list(n) for n in lista]
    if invert:
        chaves,valores = valores,chaves
    dic = {}
    for i in range(len(chaves)):
        dic[chaves[i]] = dic.get(chaves[i],0)+ valores[i]
    return dic

# Garanta que exista uma chave num dicionario para cada item de uma lista
def normalizedict(items: list):
    def normalize(dic: dict):
        for i in items:
            dic[i] = dic.get(i, 0)
        return dic
    return normalize

# Salva um objecto como string em um arquivo txt com nome = filename
def save(txt="content",filename="default"):
    with open(f'{filename}.txt', 'w') as file:
        file.write(str(txt))
    return txt


# Acha a maior 'key'de um dictionary onde os 'values' são numericos
def biggerkey(dic:dict):
    bigger = 0
    lastBiggerKey = 0
    for i,key in enumerate(dic.keys()):
        if bigger < dic[key]:
            bigger = dic[key]
            lastBiggerKey = key
    return lastBiggerKey,bigger

# Cria um dicionario a partir de uma lista de duplas,
# onde a key é a dupla[0] e o value é uma listas de duplas[1] = [...,duplas[1]]
def dictlist(lista:list):
    dic = {}
    for dupla in [list(n) for n in list(lista)]:
        dic[dupla[0]] = [*dic.get(dupla[0],[]),dupla[1]]
    return dic

# Da sort em um dictionary a partir de um dictionary
def sortbyvalue(dic,reverse=False):
    return   dict(sorted(dic.items(), key=lambda item: item[1], reverse=reverse))


#### CSV
# ##-Funcões auxiliares-##
def sanitizecsv(line: str):
    import re
    newLine = []
    for item in line:
        for quebra in re.findall(r'(\n)|(["])', item):
            item = item.replace(quebra[0], "")
            item = item.replace(quebra[1], "")
        newLine.append(item)

    return newLine