import numpy as np
import pandas as pd
import operator
from copy import deepcopy
from math import log
from anytree import Node, RenderTree
from anytree.exporter import JsonExporter
from anytree.importer import JsonImporter
import os
import json
import argparse
from anytree.dotexport import RenderTreeGraph
from sklearn.model_selection import KFold
import pickle

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
    """
    Funkcja filtruje ze zbioru litery N i S. Jej wykorzystanie w kodzie jest opcjonalne
    """
    dataTemporal = deepcopy(data)
    idxsN = [idx   for  idx, sequence in enumerate(sequences) if np.where(sequence=='N')[0].size]
    idxsS = [idx   for  idx, sequence in enumerate(sequences) if np.where(sequence=='S')[0].size]
    if not idxsS:
        indices = idxsN
    else:
        indices = np.concatenate((idxsN, idxsS))
    for i in sorted(np.unique(indices), reverse=True):
        del dataTemporal[i]
    return dataTemporal

def prepareData(data, filterNS=False):
    """
    Funkcja przygotowywuje dane do dalszego przetwarzania
    """
    dataLocal = deepcopy(data)
    dataLocal = np.reshape(a=dataLocal, newshape=(int(len(dataLocal)/2), 2))
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


def countFrequencyClasses(sequences, attributes, classes):
    """
    policz czestosc pozytywnych/negatywnych dla poszczególnych nukleodytów w poszczególnych cechach
    """
    p = [{c: 0 for c in attributes} for i in range(sequences.shape[1])]
    n = deepcopy(p)
    pAn = deepcopy(p)
    for ridx, row in enumerate(sequences):
        for cidx, column in enumerate(row):
            pAn[cidx][column] += 1
            if classes[ridx] == 1:
                p[cidx][column] += 1
            else:
                n[cidx][column] += 1
    if not p or not n:
        a = 3
    return p, n, pAn


def calculate_frequency(pAn):
    """
    p = list of dictionaries caunting positive& negative for attributes in features
    y - output (0, 1)
    """
    totalInRow = [sum(row.values()) for row in pAn] #moze dla innych danych to bedzie mialo sens ale na razie wszedzie jest stała - do optymalizacji 
    frequencies = deepcopy(pAn)
    for row, rowSum in zip(frequencies, totalInRow):
        for key in row:
            row[key] = row[key]/rowSum
    return frequencies


def singleEntropy(f, nrF):
    """
    Funkcja liczy entropie dla pojedynczego podzbioru cechy
    """
    entropy = 0
    if f > 0:
        entropy = -f*log(f, nrF)
    return entropy


def entropy(e1, e2, *rest):
    """
    liczy entropie dla cechy
    """
    args = np.concatenate(([e1, e2], rest)).astype(float) 
    fs = [arg/sum(args) for arg in args]
    return sum([singleEntropy(f, len(fs)) for f in fs])


def calculate_entropyLabel(labels):
    """
    liczy entropię dla klas
    """
    p = 0
    n = 0
    for ridx, row in enumerate(labels):
        if row == 1:
            p += 1
        else:
            n += 1

    pf = 0
    nf = 0
    count = len(labels)
    if count >0: 
        pf = float(p)/len(labels)
        nf = float(n)/len(labels)
   
    if pf == 0 or nf == 0:
       entropy = 0
    else:
       entropy = -pf*log(pf, 2)- nf*log(nf, 2)

    return entropy


def informationGain(class1, class2, feature):
    #licze entropie E(S)
    es = entropy(class1, class2) # niewydajna funkcja do optymalizacji
#     if not rest:
#         eArgsP = np.array([e1pair, e2pair]).astype(float)
#     else:
#         eArgsP = np.concatenate(([e1pair, e2pair], rest)).astype(float)
    eArgsP = np.array(feature).astype(float)
    # sumuje pozytywne i negatywne w eArgs
    eArgs = [sum(e) for e in eArgsP]
    #licze czestotliwosci 
    fs = [arg/sum(eArgs) for arg in eArgs]
    #fsAtributes = [[value/sum(attribute) for value in attribute] for attribute in eArgsP]
    
    fsAtributes = []
    for arg in eArgsP:
        summed = sum(arg)
        fp = 0
        fn = 0  
        if summed > 0:
            fp = arg[0] / summed
            fn = arg[1] / summed                
        fsAtributes.append([fp,fn])

    #licze entropie dla calosci
    entropies = [entropy(value[0], value[1]) for value in fsAtributes] 
    ids = sum([entropy*f for f, entropy in zip(fs, entropies)]) # wyglada na to ze kolejnosc acgt czy gtca nie jest wazna 
    infgain = es- ids
    return infgain

def getPairsPosNeg(p, n, attributes):
    #biore pary dobry/zly z poszczególnych cech i ich mozliwych opcji
    ePairs = []
    keys = p[0].keys()
    for featureP, featureN in zip(p, n):
        ePair = [(featureP[key], featureN[key]) for key in keys]
        ePairs.append(ePair)
    return ePairs

def calcID3(sequences, classes, attributes, indices, parentNode, atributeLabel):
    """
    Funkcja tworzy rekurencyjnie drzewo ID3
    """
    class1 = classes.count(1)
    class2 = classes.count(0)
    p, n, pAn = countFrequencyClasses(sequences, attributes, classes)
    frequencies = calculate_frequency(pAn) 
    entropyL = calculate_entropyLabel(classes)
    #licze liczebnosc klas dobre/zle (target)
    ePairs = getPairsPosNeg(p, n, attributes) #attributes nie jest używane tutaj
                                              # w frequencies parametry są policzone w kolejności ACGT, a w ePairs kolejność jest inna - GTAC - to moze byc dalej wazne...sprawdze

    # jak działa ten warunek - class1 i class2 to liczby
    if not class1 or not class2:
        if class1 == 0:
            node = Node(0, attributeLabel=atributeLabel, finalNode=True, pozneg = (0, 1), parent=parentNode)
        else:
            node = Node(1, attributeLabel=atributeLabel, finalNode=True, pozneg = (1, 0), parent=parentNode)
        return

    infGains = [informationGain(class1, class2, pair) for pair in ePairs  ]
    #NaN wrzucają się tam, gdzie danej litery nie ma. Trzeba to potem jakos lepiej ogarnac
    #informationGainR = np.nan_to_num(infGains)
    #znajdz atrybut (nukleodyt), ktory ma najwiekszy InformationGain
     
    #infGainLead = np.where(informationGainR == np.amax(informationGainR))
    maxInfGainIdx = infGains.index(max(infGains))   #jeśli jest wiecej niz jedna wartosc jest maksymalna to zwraca pierwszą z listy


    #stworz drzewo
    realValueOfPosition = indices[maxInfGainIdx]
    nodeName = str(realValueOfPosition) +':' + atributeLabel
    node = Node(nodeName, idx=realValueOfPosition, attributeLabel=atributeLabel, finalNode=False, pozneg = (class1, class2),parent=parentNode)
    #podziel dane wg s0 i znowu policz InformationGain
    
    dataA = {"sequences": [], 'y': []} 
    dataC = {"sequences": [], 'y': []}  
    dataG = {"sequences": [], 'y': []}  
    dataT = {"sequences": [], 'y': []}    
    
  
    for idx, sequence in enumerate(sequences):
        if sequence[maxInfGainIdx] == 'A':
            dataA['sequences'].append(sequence)
            dataA['y'].append(classes[idx])
            #dataA['indices'].append(indices)
        elif sequence[maxInfGainIdx] == 'C':
            dataC['sequences'].append(sequence)
            dataC['y'].append(classes[idx])
            #dataC['indices'].append(indices)
        elif sequence[maxInfGainIdx] == 'G':
            dataG['sequences'].append(sequence)
            dataG['y'].append(classes[idx])
            #dataG['indices'].append(indices)
        elif sequence[maxInfGainIdx] == 'T':
            dataT['sequences'].append(sequence)
            dataT['y'].append(classes[idx])    
        elif sequence[maxInfGainIdx] == 'S': 
            dataC['sequences'].append(sequence)
            dataC['y'].append(classes[idx]) 
            dataG['sequences'].append(sequence)
            dataG['y'].append(classes[idx])
        elif sequence[maxInfGainIdx] == 'N': 
            dataA['sequences'].append(sequence)
            dataA['y'].append(classes[idx])         
            dataC['sequences'].append(sequence)
            dataC['y'].append(classes[idx])
            dataG['sequences'].append(sequence)
            dataG['y'].append(classes[idx])         
            dataT['sequences'].append(sequence)
            dataT['y'].append(classes[idx])


    nodeData = [dataG , dataA , dataT , dataC ]
    labels = ['G','A','T','C']
    indData = 0
    for data in nodeData:
        try :
            data['sequences'] = np.delete(data['sequences'], maxInfGainIdx, 1)
            tempIndex = deepcopy(indices)  
            del tempIndex[maxInfGainIdx]
            calcID3(data['sequences'], data['y'], attributes, tempIndex, node, labels[indData]) 
        except ValueError:
            # print("cannot delete - data node is empty")
            pass
        indData +=1    

def searchFinalNode(x, node, answers):
    """
    Funkcja wyszukuje finalnych węzłów dla przykładu
    """
    currentAttribute = x[node.idx]
    children = node.children
    newNodes = []
    for chn in children:
            if currentAttribute==chn.attributeLabel:
                newNodes.append(chn)
            elif currentAttribute=='S':
                for chInner in children:
                    if chnInner.attributeLabel== 'G' or chnInner.attributeLabel== 'C':
                        newNodes.append(chnInner)
            elif currentAttribute=='N':
                for chInner in children:
                    newNodes.append(chn)
    if not newNodes:
        pozneg = np.zeros(shape=(1, 2))
        for chn in children:
            pozneg[0, 0] +=chn.pozneg[0]
            pozneg[0, 1] +=chn.pozneg[1]
        if pozneg[0, 0]>pozneg[0, 1]:
            answers.append(1)
        else:
            answers.append(0)
    
    for n in newNodes:
        if n.finalNode == True:
            answers.append(n.name)
        else:
            searchFinalNode(x, n, answers)
        
def predictSingle(x, root):
    """
    Funkcja znajduje odpowiedz drzewa dla podanego przykladu
    """
    node = root.children[0]
    answers = []
    searchFinalNode(x, node, answers)
                
    return answers


def predictBatch(X, root):
    """
    Funkcja znajduje odpowiedzi drzewa dla całego zbioru przykladow
    """
    return [predictSingle(i, root) for i in X]

if __name__== "__main__":
    parser = argparse.ArgumentParser("Program to make id3 tree on DNA data")
    parser.add_argument('-m', '--mode', type=str, choices=["train", "pred", 'kValid'], help='choose mode of the program')
    parser.add_argument('-s', '--source', type=str,  help='Path to the data source')
    parser.add_argument('-ns', '--nsInclude', type=bool,  default=True, help='Specify if you want to include sequences with N and S in data processing')
    parser.add_argument('-tN', '--treeName', type=str,  help='name of the tree')
    parser.add_argument('-tS', '--treeSource', type=str,  default=None, help='path to the tree, on wchih script should run validation')
    

    args = parser.parse_args()

    mode = args.mode

    filepath = os.path.normpath(args.source)

    treeName = args.treeName
    treeSource = args.treeSource

    

    data = loadData(filepath)[1:]
    cutNr = int(data[0])
    df = prepareData(data, filterNS=args.nsInclude)
    classesOrigin = list(df.y)
    attributes = list(set("".join([i for i in df.seq])))
    sequencesOrigin = np.array([list(i) for i in df.seq])
    if mode == 'kValid':
        #walidacja krzyzowa
        kf = KFold(n_splits=10, shuffle=True)
        idxs = [(trainIdxs, testIdxs) for trainIdxs, testIdxs in kf.split(classesOrigin)]
    elif mode == 'train' or mode == 'pred':
        idxs = [0]

    if mode == 'train':
        classesTrain = classesOrigin
        sequencesTrain = sequencesOrigin
    elif mode == 'pred':
        classesVal = classesOrigin
        sequencesVal = sequencesOrigin
    if mode == 'pred' or mode == 'kValid':
        average = []
    for portion1 in idxs:
        if mode == 'kValid':
            sequencesTrain = [sequencesOrigin[idx] for idx in portion1[0]]
            sequencesVal = [sequencesOrigin[idx] for idx in portion1[1]]
            classesTrain = [classesOrigin[idx] for idx in portion1[0]]
            classesVal = [classesOrigin[idx] for idx in portion1[1]]
        


            
        if mode == 'kValid' or mode == 'train':
            sequences = np.array(sequencesTrain)
            classes = classesTrain
            indices = [*range(0, sequences.shape[1])]
            root = Node('root', finalNode=False)
            #stworz drzewo ID3
            calcID3(sequences, classes, attributes,indices, root, 'root')

        if mode == 'train':
            # export tree to json file
            exporter = JsonExporter()
            with open(treeName+'.json', 'w') as outfile:
                json.dump(exporter.export(root), outfile)
            # export visual version of the tree
            fileTreeLog= open(treeName+"Picture.txt","w+")
            for row in RenderTree(root):
                fileTreeLog.write("%s%s\n" % (row.pre, row.node.name))
        
            for pre, fill, node in RenderTree(root):
                fileTreeLog.write("%s%s\n" % (pre, node.name))

            fileTreeLog.close()  
        
        if mode == 'pred' or mode == 'kValid':

            sequences = sequencesVal
            ys = classesVal

            #import tree from json file
            if mode == 'pred':
                with open(treeSource) as infile:
                    jsonTree = json.load(infile)
                importer = JsonImporter()
                root = importer.import_(jsonTree)
            #przeprowadz predykcje na zbiorze
            y_preds = predictBatch(sequences, root)
            y_preds = [y_pred[0] if len(y_pred)==1 else int(np.median(y_pred)) for y_pred in y_preds]
            answers = {'good': 0, 'bad': 0}
            for y, y_pred in zip(ys, y_preds):
                if y == y_pred:
                    answers['good'] += 1
                else:
                    answers['bad'] += 1
            print("Wynik predykcji: "+str(answers['good']/(answers['good']+answers['bad'])))
    
            # print(RenderTree(root)) 
            average.append(answers['good']/(answers['good']+answers['bad']))
        
    if mode == 'kValid':
        print('srednio: ', np.mean(average))
      


    

    