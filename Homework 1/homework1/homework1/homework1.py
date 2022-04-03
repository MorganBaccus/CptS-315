import operator


baskets = list()
f = open('./browsingData.txt', 'r')
lines = f.read().splitlines()
for line in lines:
    baskets.append(set(line.split(" ")))
f.close()

items = dict()
for basket in baskets:
    for item in basket:
        items[item] = items.get(item, 0) + 1
items.pop('', None)

freqItems = dict()
for key in items:
    value = items[key]
    if (value >= 100):
        freqItems[key] = value
        
candidates = dict()
for basket in baskets:
    for index_i, item_i in enumerate(basket):
        if item_i in freqItems:
            for index_j, item_j in enumerate(basket):
                if index_j > index_i:
                    if item_j in freqItems:
                        if item_i > item_j:
                            candidates[item_i, item_j] = candidates.get((item_i, item_j), 0) + 1
                        else:
                            candidates[item_j, item_i] = candidates.get((item_j, item_i), 0) + 1

freqCandidates = dict()
for key in candidates:
    value = candidates[key]
    if (value >= 100):
        freqCandidates[key] = value

candidates3 = dict()
for basket in baskets:
    for index_i, item_i in enumerate(basket):
        if item_i in freqItems:
            for index_j, item_j in enumerate(basket):
                if index_j > index_i:
                    if item_j in freqItems:
                        if (item_i,item_j) in freqCandidates or (item_j,item_i) in freqCandidates: 
                            for index_k, item_k in enumerate(basket):
                                if index_k > index_j:
                                    if item_k in freqItems:
                                        if (item_i,item_k) in freqCandidates or (item_k,item_i) in freqCandidates:
                                            if (item_j,item_k) in freqCandidates or (item_k,item_j) in freqCandidates:
                                                if item_i > item_j:
                                                    if item_k > item_i:
                                                        candidates3[item_k, item_i, item_j] = candidates3.get((item_k, item_i, item_j), 0) + 1
                                                    elif item_k > item_j:
                                                        candidates3[item_i, item_k, item_j] = candidates3.get((item_i, item_k, item_j), 0) + 1
                                                    else:
                                                        candidates3[item_i, item_j, item_k] = candidates3.get((item_i, item_j, item_k), 0) + 1
                                                else:
                                                    if item_k > item_j:
                                                        candidates3[item_k, item_j, item_i] = candidates3.get((item_k, item_j, item_i), 0) + 1
                                                    elif item_k > item_i:
                                                        candidates3[item_j, item_k, item_i] = candidates3.get((item_j, item_k, item_i), 0) + 1
                                                    else:
                                                        candidates3[item_j, item_i, item_k] = candidates3.get((item_j, item_i, item_k), 0) + 1

freqCandidates3 = dict()
for key in candidates3:
    value = candidates3[key]
    if (value >= 100):
        freqCandidates3[key] = value      
                                                                                                                                                                                               
rules = dict()
for key in freqCandidates:
    rules[key[0], key[1]] = freqCandidates[key]/freqItems[key[0]]
    rules[key[1], key[0]] = freqCandidates[key]/freqItems[key[1]]
rules3 = dict()
for key in freqCandidates3:
    if (key[0], key[1]) in freqCandidates:
        rules3[key[0], key[1], key[2]] = freqCandidates3[key]/freqCandidates[key[0], key[1]]
    else:
        rules3[key[0], key[1], key[2]] = freqCandidates3[key]/freqCandidates[key[1], key[0]]
    if (key[0], key[2]) in freqCandidates:    
        rules3[key[0], key[2], key[1]] = freqCandidates3[key]/freqCandidates[key[0], key[2]]
    else:
        rules3[key[0], key[2], key[1]] = freqCandidates3[key]/freqCandidates[key[2], key[0]]
    if (key[1], key[2]) in freqCandidates:
        rules3[key[1], key[2], key[0]] = freqCandidates3[key]/freqCandidates[key[1], key[2]]
    else:
        rules3[key[1], key[2], key[0]] = freqCandidates3[key]/freqCandidates[key[2], key[1]]
        
sortedRules = sorted(rules.items(), key=operator.itemgetter(1), reverse=True)
sortedRules3 = sorted(rules3.items(), key=operator.itemgetter(1), reverse=True)

top = list()
tie = list()
thresholdConfidence = sortedRules[5-1][1] 
for rule in sortedRules:
    if (rule[1] > thresholdConfidence):
        top.append(rule)
    elif (rule[1] == thresholdConfidence):
        tie.append(rule)
    else:
        break
tie.sort(key=operator.itemgetter(0))
top = top + tie[0:(5-len(top))]    

top3 = list()
tie3 = list()
thresholdConfidence = sortedRules3[5-1][1] 
for rule in sortedRules3:
    if (rule[1] > thresholdConfidence):
        top3.append(rule)
    elif (rule[1] == thresholdConfidence):
        tie3.append(rule)
    else:
        break
tie3.sort(key=operator.itemgetter(0))
top3 = top3 + tie3[0:(5-len(top3))] 
 
f = open('./outfile.txt', 'w')
f.write('OUTPUT A' + '\n')
for rule in top:
    f.write(rule[0][0] + ' ' + rule[0][1] + ' ' + '{:.4f}'.format(rule[1]) + '\n')
f.write('OUTPUT B' + '\n')
for rule in top3:
    f.write(rule[0][0] + ' ' + rule[0][1] + ' ' + rule[0][2] + ' ' + '{:.4f}'.format(rule[1]) + '\n')
f.close()