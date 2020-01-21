import numpy as np
import pandas as pd
import operator
from copy import deepcopy
from math import log
from anytree import Node, RenderTree
import os

def loadData(filepath):
    data = []
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            data.append(line.strip())
            line = fp.readline()
            cnt += 1
    return data

def filterLetters(data, sequences):
    dataTemporal = deepcopy(data)
    idxsN = [idx   for  idx, sequence in enumerate(sequences) if np.where(sequence=='N')[0].size]
    idxsS = [idx   for  idx, sequence in enumerate(sequences) if np.where(sequence=='S')[0].size]
    indices = np.concatenate((idxsN, idxsS))
    for i in sorted(np.unique(indices), reverse=True):
        del dataTemporal[i]
    return dataTemporal

def prepareData(data, filterNS=False):
    dataLocal = deepcopy(data)
    dataLocal = np.reshape(a=dataLocal, newshape=(int(11576/2), 2))
    df = pd.DataFrame(dataLocal, columns=['y', "seq"])
    if filterNS:
        sequences = np.array([list(i) for i in df.seq])
        dataF = filterLetters(list(dataLocal), sequences)
        dfF = pd.DataFrame(dataF, columns=['y', "seq"])
        dfF.y = pd.to_numeric(dfF.y)
        return dfF
    else:
        df.y = pd.to_numeric(df.y)
        return df


def countFrequencyClasses(sequences, attributes):
    """
    policz czestosc pozytywnych/negatywnych dla poszczególnych nukleodytów w poszczególnych cechach
    """
    p = [{c: 0 for c in attributes} for i in range(sequences.shape[1])]
    n = deepcopy(p)
    pAn = deepcopy(p)
    for ridx, row in enumerate(sequences):
        for cidx, column in enumerate(row):
            pAn[cidx][column] += 1
            if df.y[ridx] == 1:
                p[cidx][column] += 1
            else:
                n[cidx][column] += 1
    return p, n, pAn


def calculate_frequency(pAn):
    """
    p = list of dictionaries caunting positive& negative for attributes in features
    y - output (0, 1)
    """
    totalInRow = [sum(row.values()) for row in pAn]
    frequencies = deepcopy(pAn)
    for row, rowSum in zip(frequencies, totalInRow):
        for key in row:
            row[key] = row[key]/rowSum
    return frequencies


def singleEntropy(f, nrF):
    """
    Funkcja liczy entropie dla pojedynczego podzbioru cechy
    """
    return -f*log(f, nrF)
def entropy(e1, e2, *rest):
    """
    liczy entropie dla cechy
    """
    args = np.concatenate(([e1, e2], rest)).astype(float) 
    fs = [arg/sum(args) for arg in args]
    return sum([singleEntropy(f, len(fs)) for f in fs])


def calculate_entropyLabel(labels):
    p = 0
    n = 0
    for ridx, row in enumerate(labels):
        if row == 1:
            p += 1
        else:
            n += 1
    pf = p/len(labels)
    nf = n/len(labels)
    entropy = -pf*log(pf, 2)-nf*log(nf, 2)
    return entropy


def informationGain(class1, class2, feature):
    #licze entropie E(S)
    es = entropy(class1, class2)
#     if not rest:
#         eArgsP = np.array([e1pair, e2pair]).astype(float)
#     else:
#         eArgsP = np.concatenate(([e1pair, e2pair], rest)).astype(float)
    eArgsP = np.array(feature).astype(float)
    # sumuje pozytywne i negatywne w eArgs
    eArgs = [sum(e) for e in eArgsP]
    #licze czestotliwosci 
    fs = [arg/sum(eArgs) for arg in eArgs]
    fsAtributes = [[value/sum(attribute) for value in attribute] for attribute in eArgsP]
    #licze entropie dla calosci
    entropies = [entropy(value[0], value[1]) for value in fsAtributes]
    return es-sum([entropy*f for f, entropy in zip(fs, entropies)])

def getPairsPosNeg(p, n):
    #biore pary pobry/zly z poszczególnych cech i ich mozliwych opcji
    ePairs = []
    keys = p[0].keys()
    for featureP, featureN in zip(p, n):
        ePair = [(featureP[key], featureN[key]) for key in keys]
        ePairs.append(ePair)
    return ePairs



if __name__== "__main__":
    
    fileAbsPath = os.path.realpath(__file__) 
    absPath = fileAbsPath.rsplit('/',1)[0]

    filepath =  absPath + '/data/spliceATrainKIS.dat'
    data = loadData(filepath)[1:]
    cutNr = int(data[0])
    df = prepareData(data, filterNS=True)

    classes = list(df.y)
    attributes = set("".join([i for i in df.seq]))
    sequences = np.array([list(i) for i in df.seq])


    ##############################################################
    nodes = []
    p, n, pAn = countFrequencyClasses(sequences, attributes)
    frequencies = calculate_frequency(pAn)
    entropyL = calculate_entropyLabel(classes)
    #licze liczebnosc klas dobre/zle (target)
    class1 = classes.count(1)
    class2 = classes.count(0)
    ePairs = getPairsPosNeg(p, n)
    infGains = [informationGain(class1, class2, pair) for pair in ePairs  ]
    #NaN wrzucają się tam, gdzie danej litery nie ma. Trzeba to potem jakos lepiej ogarnac
    informationGainR = np.nan_to_num(infGains)
    #znajdz atrybut (nukleodyt), ktory ma najwiekszy InformationGain
    
    infGainLead = np.where(informationGainR == np.amax(informationGainR))
    #stworz drzewo
    if not nodes:
        nodes.append(Node(infGainLead, sequences=sequences, classes=classes))
    else:
        nodes.append(Node(infGainLead, sequences=sequences, classes=classes), parent=nodes[-1])
    #podziel dane wg s0 i znowu policz InformationGain
    keys = p[0].keys()
    dataA = {"sequences": [], 'y': []} 
    dataC = {"sequences": [], 'y': []} 
    dataG = {"sequences": [], 'y': []} 
    dataT = {"sequences": [], 'y': []} 
    for idx, sequence in enumerate(sequences):
        if sequence[infGainLead] == 'A':
            dataA['sequences'].append(sequence)
            dataA['y'].append(df.y[idx])
        elif sequence[infGainLead] == 'C':
            dataC['sequences'].append(sequence)
            dataC['y'].append(df.y[idx])
        elif sequence[infGainLead] == 'G':
            dataG['sequences'].append(sequence)
            dataG['y'].append(df.y[idx])
        elif sequence[infGainLead] == 'T':
            dataT['sequences'].append(sequence)
            dataT['y'].append(df.y[idx])
        #doprowadzić dane do stanu na poczatku

        ########################################################################
    




    

    