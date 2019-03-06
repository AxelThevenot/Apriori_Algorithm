import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# region loading dataset
def loadDataSet():
    """ To load the data set (change the funnction to the case)
    :return: dataset
    """
    return [['Apple', 'Corn', 'Milk', 'Orange Juice', 'Yogurt', 'Flour'],
           ['Water', 'Corn', 'Milk', 'Orange Juice', 'Yogurt', 'Flour'],
           ['Apple', 'Apple', 'Orange Juice', 'Flour'],
           ['Apple', 'Sugar', 'Corn', 'Orange Juice', 'Yogurt'],
           ['Water', 'Corn', 'Corn', 'Orange Juice', 'Ice cream', 'Yogurt']]

def loadMushroomDataSet(path):
    """
    return the dataset (list) of mushroom from path file
    :param path: path of mushroom dataset
    :return: mushroom dataset
    """
    mushroomDataset = None
    try :
        mushroomDataset = [line.split() for line in open(path).readlines()]
    except Exception as e:
        print(e)
    finally:
        return mushroomDataset

# endregion

# region Apriori algorithm
def Apriori(dataset, minSupport):
    """
    Apriori algorithm
    :param transaction: dataset of transaction
    :param minSupport: support value we are interested in
    :return: array for all Lk
    """
    C1 = Candidate_1(dataset)
    # convert the dataset to set of transcation if needed
    dataset = list(map(set, dataset))
    L1, supportData = scanDataSet(dataset, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = Candidate_k(L[k - 2], k)
        Lk, supportData_k = scanDataSet(dataset, Ck, minSupport)
        supportData.update(supportData_k)
        L.append(Lk)
        k += 1

    return L, supportData

def Candidate_1(dataSet):
    """
    To create the first array of candidate
    :param dataSet: dataset
    :return: return te fisrt layer of candidate
    """
    C1 = []
    # Candidates in the first layer are all the single items
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    # we frozen the itemset to use it as a key dict
    return list(map(frozenset, C1))

def Candidate_k(Lkminus1, k):
    """
    Create C[k] from L[k-1] and k
    :param Lkminus1: the frequents itemsets (k-1)-sized
    :param k: k if the size of itemsets
    :return: Ck
    """
    Ck = []
    lenLk = len(Lkminus1)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lkminus1[i])[:k-2]
            L2 = list(Lkminus1[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2: #if first k-2 elements are equal
                Ck.append(Lkminus1[i] | Lkminus1[j]) #set union
    return Ck

def scanDataSet(dataset, Ck, minSupport):
    """
    Function to generate Lk from Ck
    :param D: dataset
    :param Ck: candidates of k items
    :param minSupport: value of support value interested in
    :return: return Lk, dict of support values
    """
    # we create a dict of value to count the amount of transaction supported by candidate
    subsetCounter = {}
    for transaction in dataset:
        for candidate in Ck:
            if candidate.issubset(transaction):
                if not candidate in subsetCounter: subsetCounter[candidate]=1
                else: subsetCounter[candidate] += 1
    number_items = float(len(dataset))
    Lk = []
    supportData = {}
    # candidate in Ck are in Lk if they support minimum value
    for subsetCandidate in subsetCounter:
        support = subsetCounter[subsetCandidate] / number_items
        if support >= minSupport:
            Lk.insert(0, subsetCandidate)
        supportData[subsetCandidate] = support
    return Lk, supportData
# endregion

# region generating rules from confidence
def generateRules(L, supportData, minConf=0.7):
    """
    Generate rules of (k)-itemset implied by (k-1)-itemset
    :param L: list of all Lk
    :param supportData: dict of support data value
    :param minConf: minimum confidence of rule interested in
    :return: list of all the rules implied by L
    """
    rules = []
    # only get the sets with two or more items
    for i in range(1, len(L)):
        for freqentSet in L[i]:
            H1 = [frozenset([item]) for item in freqentSet]
            if (i > 1):
                rulesFromConsequence(freqentSet, H1, supportData, rules, minConf)
            else:
                calculateConfidence(freqentSet, H1, supportData, rules, minConf)
    return rules

def calculateConfidence(freqentSet, H, supportData, rules, minConf=0.7):
    """
    calculates the confidence of the rule and then find out the which rules meet the minimum confidence.
    :param freqSet: frequent itemset
    :param H: list of items that could be on the right-hand side of a rule
    :param supportData: supportData from Apriori algorithm
    :param rules: list of all the rules implied by L
    :param minConf: minimum confidence interested in
    :return: all the rules implied by H
    """
    prunedH = [] # create new list to return
    for conseq in H:
        # check the confidence is supported
        conf = supportData[freqentSet]/supportData[freqentSet-conseq] # calculate confidence
        if conf >= minConf:
            rules.append((freqentSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConsequence(freqSet, H, supportData, rules, minConf=0.7):
    """
    generate more association from the inital dataset merging it
    :param freqSet: frequent itemset
    :param H: list of items that could be on the right-hand side of a rule
    :param supportData: supportData from Apriori algorithm
    :param rules: list of all the rules implied by L
    :param minConf: minimum confidence interested in
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)): # try further merging
        Hmp1 = Candidate_k(H, m+1)# create Hm+1 new candidates
        Hmp1 = calculateConfidence(freqSet, Hmp1, supportData, rules, minConf)
        if (len(Hmp1) > 1):    # need at least two sets to merge
            rulesFromConsequence(freqSet, Hmp1, supportData, rules, minConf)
# endregion

# region creating a graph
def createGraph(L, support):
    """
    create a graph (nodes) to represent links between transactions
    :param L: L given by apriori algorithm
    :param support: support dataset given by apriori algorithm
    """

    # for each layer of partition ordered by size
    G = nx.Graph()
    for size in range(1, len(L)):
        # for each partition in this layer
        for _, partition in enumerate(L[size]):
            partition_list = list(partition)
            # each partition is linked to every size-1 partition with same items (without one)
            G.add_node(str(partition_list), support=support[frozenset(partition)])
            for i in range(size + 1):
                from_partition = partition_list[:i] + partition_list[i + 1:]
                if str(from_partition) not in G.node:
                    G.add_node(str(from_partition), support=support[frozenset(from_partition)])
                G.add_edge(str(partition_list), str(from_partition),
                           conf= support[frozenset(partition)] / support[frozenset(from_partition)] )
    # position of nodes organised in layer of item number
    number_node_layer = [len(Lk) for _, Lk in enumerate(L)]
    size_plot = max(number_node_layer)
    count_node_positionned = [0]*len(number_node_layer)
    position_list = []
    for _, n in enumerate(G.nodes):
        l = len(n.split(","))
        position_list.append((l, count_node_positionned[l-1] * size_plot / number_node_layer[l-1]))
        count_node_positionned[l-1] += 1
    #nx.draw_networkx_edge_labels(G, pos=nx.shell_layout(G))
    # to return an hexa char
    def hexa_char(number):
        number = int(number)
        if number > 9:
            return chr(87 + number)
        return number
    # coloring each edge by its confidence
    # coloring each node by its frequency (support)

    nx.draw_networkx(G, font_size=8, pos=dict(zip(G,position_list)),
                     edge_color=['#'+str(hexa_char(16-(G[e][edge]['conf']*2**4)))*6
                                 for e, edge in G.edges()],
                     node_color = ['#00'+str(hexa_char(16-G.nodes[node]['support']*12))*4
                                 for node in G.node()])
    #uncomment the line below to display this plot
    # plt.show()

#endregion

#region creating rule heatmap
def createRuleHeatmap(rules):
    """
    create a Heatmap to represent rules
    :param rules: rules given by apriori rules algorithm
    """
    # we create a list to recover the rules data
    # k-th rule say that 'list_rules[k][1] is implied by list_rules[k][0] with a confidence of list_rules[k][2]
    list_rules = [list((str((list(rule[0]))), str((list(rule[1]))), rule[2]))
                  for _, rule in enumerate(rules)]
    # we create a column made by all involving items removing duplicated
    column = list(set([r[0] for _, r in enumerate(list_rules)]))
    # we create a column made by all involved items removing duplicated
    row = list(set([r[1] for _, r in enumerate(list_rules)]))

    # and we create a matrix of confidence between each items given by rules
    rules_confidence = np.zeros((len(row), len(column)))
    for _, rule in enumerate(rules):
        rules_confidence[row.index(str(list(rule[1]))), column.index(str(list(rule[0])))] = rule[2]

    # finally create the plot (HeatMap)
    fig, axis = plt.subplots()  # il me semble que c'est une bonne habitude de faire supbplots
    heatmap = axis.pcolor(rules_confidence, cmap=plt.cm.Blues)  # heatmap contient les valeurs

    axis.set_yticks(np.arange(rules_confidence.shape[0]) + 0.5, minor=False)
    axis.set_xticks(np.arange(rules_confidence.shape[1]) + 0.5, minor=False)

    axis.invert_yaxis()

    axis.set_yticklabels(row, minor=False)
    axis.set_xticklabels(column, minor=False)


    plt.colorbar(heatmap)
    plt.show()

#endregion


# region using algorithm sample
# variables
minSupport = 0.5
minConfidence = 0.7
dataset = loadDataSet()

# apriori algorithm
L, supportData = Apriori(dataset , minSupport)
rules = generateRules(L, supportData, minConfidence)

# display
createGraph(L, supportData)
createRuleHeatmap(rules)
# endregion

