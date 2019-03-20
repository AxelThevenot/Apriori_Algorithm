## Algorithm principle

The Apriori algorithm is used to recognize item sets, which frequently appear in a data set and infer a categorization from it. It means that if a set is frequent, its subsets are frequent too. In easy words, if {Milk, Apple} is a frequent item set, {Milk} and {Apple} are frequent item sets too.

## Relationships association

Now imagine running the algorithm for a list of transactions, which can be composed of 15 different items. Checking the frequency of all the possible combinations of those 15 items (ca. 1 300 billion item sets) would be a time-consuming task and even impossible if there are more items. That is why the Apriori algorithm needs a minimum support (between 0 and 1). If the occurrence of an item set is lower than the minimum support, its supersets can not have an occurrence greater than this minimum support. The Apriori algoritmh works on this principle. 


The Apriori algotrithm aims at identifying relationships between the item(s) combinations. 

Two cases of association can be found : 
* Frequent item sets are in the collection of every item set that occurs more than the minimum support
* Collection of association rule as "Itemset --> SuperItemset". A rule is in this collection if the confidence is greater than the minimum confidence given. The confidence is defined by support(Itemset | SuperItemset)/ support(Itemset)
Looking for hidden relationships in large data sets is known as association analysis or association rule learning. 

## Dataset

In this case we will use a data set of transactions for the sake of understanding. 

```python 
def loadDataSet():
    """ To load the dataset (change the function to the case)
    :return: dataset
    """
    return [['Apple', 'Corn', 'Milk', 'Orange Juice', 'Yogurt', 'Flour'],
           ['Water', 'Corn', 'Milk', 'Orange Juice', 'Yogurt', 'Flour'],
           ['Apple', 'Apple', 'Orange Juice', 'Flour'],
           ['Apple', 'Sugar', 'Corn', 'Orange Juice', 'Yogurt'],
           ['Water', 'Corn', 'Corn', 'Orange Juice', 'Ice cream', 'Yogurt']
```
## Pseudo Code

```
Load the dataset and give a minimum support

k=1

Create a list Ck of candidate itemsets of length k

Scan the dataset to return Lk which is composed of the itemsets of each frequent itemset in Ck

Repeat the last 2 lines with k+1 while Lk is not an empty set.
```

# Let's start with python

The first step is to create a function to return the first array of candidate C1 composed of every single item in the data set.

```python 
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
```
The second step is to create the function to return the frequent item sets in the data set. This function takes three arguments: a dataset, the list of  a list of candidate sets Ck, and the minimum support of interest. This is the function we will use to generate L1 from C1. This function returns also a dictionary with support values.

```python 
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
```

We can try it now

```python 
dataset = loadDataSet()
minSupport = 0.5
C1 = Candidate_1(dataset)
print("C1 : ", C1)
L1, support_1 = scanDataSet(dataset, C1, minSupport)
print("L1 : ", L1)
print("support 1 : ", support_1)
```
We can see that the items {Milk}, {water}, {Sugar} and {Ice cream} have their support value lower than the minimum support so there are not frequent item sets in L1

[Output]

```
C1 :  [frozenset({'Apple'}), frozenset({'Corn'}), frozenset({'Milk'}), frozenset({'Orange Juice'}), frozenset({'Yogurt'}), frozenset({'Flour'}), frozenset({'Water'}), frozenset({'Sugar'}), frozenset({'Ice cream'})]
L1 :  [frozenset({'Flour'}), frozenset({'Yogurt'}), frozenset({'Orange Juice'}), frozenset({'Corn'}), frozenset({'Apple'})]
support 1 :  {frozenset({'Apple'}): 0.6, frozenset({'Corn'}): 0.8, frozenset({'Milk'}): 0.4, frozenset({'Orange Juice'}): 1.0, frozenset({'Yogurt'}): 0.8, frozenset({'Flour'}): 0.6, frozenset({'Water'}): 0.4, frozenset({'Sugar'}): 0.2, frozenset({'Ice cream'}): 0.2}
```
Then, we create the function to create k-sized candidates from Lk-1. This function takes two arguments, the frequent item sets of size k-1 Lkminus1 and the size of the new item sets k. It will create a set of candidates composed of the union of each item sets of Lkminus1 and of each item in these item sets.

```python 
def Candidate_k(Lkminus1, k):
    """
    Create C[k] from L[k-1] and k
    :param Lkminus1: the frequent itemsets (k-1)-sized
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
```

All the functions needed to run the Apriori function are now ready. We can define the Apriori function that takes two arguments, the dataset and the minimum support of interest. This function will return an array of all Lk and the dictionnary of every candidate's support. 

```python 
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
``` 
We can try the algorithm 
```python 
dataset = loadDataSet()
minSupport = 0.5
L, supportData = Apriori(dataset, minSupport)
for k, k_sized_itemsets in enumerate(L):
    print("frequent itemsets of size {0} : {1}".format(k+1, k_sized_itemsets))
```

[Output]
```
frequent itemsets of size 1 : [frozenset({'Flour'}), frozenset({'Yogurt'}), frozenset({'Orange Juice'}), frozenset({'Corn'}), frozenset({'Apple'})]
frequent itemsets of size 2 : [frozenset({'Orange Juice', 'Apple'}), frozenset({'Orange Juice', 'Corn'}), frozenset({'Corn', 'Yogurt'}), frozenset({'Orange Juice', 'Yogurt'}), frozenset({'Orange Juice', 'Flour'})]
frequent itemsets of size 3 : [frozenset({'Orange Juice', 'Yogurt', 'Corn'})]
frequent itemsets of size 4 : []
```


## Generating rules 

To find association rules, we first start with a frequent item set. We know this set of items is unique, but we want to see if there is anything else we can get out of these items. One item or one set of items can imply another item.
As a reminder : Itemset --> SuperItemset. A rule is in this collection if the confidence is greater than the minimum confidence given. The confidence is defined by support(Itemset | SuperItemset)/ support(Itemset)

generateRules(), is the main command, which calls the other two.

The generateRules() function takes three inputs: a list of frequent item sets, a dictionary of support data for those item sets, and a minimum confidence we are interested in. It will generate a list of rules with confidence values that we can sort later.
```python
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
```
As we can see there are two functions calculateConfidence() and rulesFromConsequence() called by generateRules(). 
calculateConfidence(), which calculates the confidence of the rule and then finds out the rules meeting the minimum confidence. It takes five arguments, a frequent item set, the list H of item sets that could be the consequence of the frequent item set, the support dictionary given by the Apriori algorithm, the list of rules we want to implement and the minimum confidence we are interested in.   

```python
def calculateConfidence(freqentSet, H, supportData, rules, minConf=0.7):
    """
    calculates the confidence of the rule and then find out the which rules meet the minimum confidence.
    :param freqSet: frequent itemset
    :param H: list of items that could be on the right-hand side of a rule
    :param supportData: supportData from Apriori algorithm
    :param rules: list of all the rules implied by L
    :param minConf: minimum confidence of interest
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
```
generateRules(), which generates new associations by merging the existing ones recursively. It takes the same five arguments that the previous function calculateConfidence()

```python 
def rulesFromConsequence(freqSet, H, supportData, rules, minConf=0.7):
    """
    generate more association from the inital dataset merging it
    :param freqSet: frequent itemset
    :param H: list of items that could be on the right-hand side of a rule
    :param supportData: supportData from Apriori algorithm
    :param rules: list of all the rules implied by L
    :param minConf: minimum confidence of interest in
    """
    m = len(H[0])
    if (len(freqSet) > (m + 1)): # try further merging
        Hmp1 = Candidate_k(H, m+1)# create Hm+1 new candidates
        Hmp1 = calculateConfidence(freqSet, Hmp1, supportData, rules, minConf)
        if (len(Hmp1) > 1):    # need at least two sets to merge
            rulesFromConsequence(freqSet, Hmp1, supportData, rules, minConf)
```
Each function is now written for the Apriori algorithm. Before trying to render the output, the new functions for the associations rules must be tested.

```python 
dataset = loadDataSet()
minSupport = 0.5
minConfidence = 0.7
L, supportData = Apriori(dataset, minSupport)
rules = generateRules(L, supportData, minConfidence)
for _, rule in enumerate(rules):
    print("{0} --> {1} with a confidence of {2}".format(list(rule[1]), list(rule[0]), rule[2]))
```
[Output]
```
['Orange Juice'] --> ['Apple'] with a confidence of 1.0
['Corn'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice'] --> ['Corn'] with a confidence of 1.0
['Corn'] --> ['Yogurt'] with a confidence of 1.0
['Yogurt'] --> ['Corn'] with a confidence of 1.0
['Orange Juice'] --> ['Yogurt'] with a confidence of 1.0
['Yogurt'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice'] --> ['Flour'] with a confidence of 1.0
['Corn', 'Orange Juice'] --> ['Yogurt'] with a confidence of 1.0
['Corn', 'Yogurt'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice', 'Yogurt'] --> ['Corn'] with a confidence of 1.0
```


## Rendering

We want now to display our results from our algorithm. I personally choose to make two different plots : 
* A graph of nodes to display the frequent item sets
* A heatmap to display rules 

We need to import numpy, networkx especially for the graph and matplotlib.pyplot to plot these rendering

```python 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
```

Graph creation : the function takes two arguments, the frequent item sets L and the support dictionary. For a better display I choose to color each node according to its frequency (support) and each edge according to its confidence. The darkest it is the greatest value it is .


```python 
def createGraph(L, support):
    """
    create a graph (nodes) to represent links between transactions
    :param L: L given by Apriori algorithm
    :param support: support dataset given by Apriori algorithm
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
    plt.show()
```

Here we can see the render as a png file. We can interpret this graph by : 
* {Yogurt}, {Orange Juice} and {Yogurt, Orange Juice} are often bought
* {Apple} is not often bought
* {Apple, Orange juice} often implied the {Apple} buying
* {Apple, Orange juice} not often implied the {Orange juice} buying

![](/src/graph.png)

As we can see it is pretty hard to discuss the rules with this graph, so we will create a heatmap of "what item sets (column)  involve what item (row)".
It works with the same principle as the previous graph. This function takes only the rules as argument.

```python 
def createRuleHeatmap(rules):
    """
    create a Heatmap to represent rules
    :param rules: rules given by Apriori rules algorithm
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
    fig, axis = plt.subplots()  
    heatmap = axis.pcolor(rules_confidence, cmap=plt.cm.Blues)  # heatmap contient les valeurs

    axis.set_yticks(np.arange(rules_confidence.shape[0]) + 0.5, minor=False)
    axis.set_xticks(np.arange(rules_confidence.shape[1]) + 0.5, minor=False)

    axis.invert_yaxis()

    axis.set_yticklabels(row, minor=False)
    axis.set_xticklabels(column, minor=False)


    plt.colorbar(heatmap)
    plt.show()
```

As a reminder the rules were :

```
['Orange Juice'] --> ['Apple'] with a confidence of 1.0
['Corn'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice'] --> ['Corn'] with a confidence of 1.0
['Corn'] --> ['Yogurt'] with a confidence of 1.0
['Yogurt'] --> ['Corn'] with a confidence of 1.0
['Orange Juice'] --> ['Yogurt'] with a confidence of 1.0
['Yogurt'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice'] --> ['Flour'] with a confidence of 1.0
['Corn', 'Orange Juice'] --> ['Yogurt'] with a confidence of 1.0
['Corn', 'Yogurt'] --> ['Orange Juice'] with a confidence of 0.8
['Orange Juice', 'Yogurt'] --> ['Corn'] with a confidence of 1.0
```

Now the heatmap below shows the same confidences with colors.


![](/src/heatmap.png)
