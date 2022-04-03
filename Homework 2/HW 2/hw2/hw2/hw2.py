#hw2.py
import operator
import numpy as np
import pandas as pd
import csv
from itertools import islice


class IICFilter():
    def __init__(self, infile, outfile="output.txt", n=5, userTop=5):
        self.file = infile
        self.outfile = outfile
        self.neighborhood = n
        self.userTop = userTop

    def run(self):
        print("The program is running. Please wait.\n")
        movieRatings = self.readMovies
        maxMId, maxUId, mIds, userIds = self.uniqueColumns(movieRatings)
        nUniques, mt = self.evalRatings(maxMId, maxUId, mIds, movieRatings, userIds)
        scores = self.compSimilarity(nUniques, mIds, mt)
        neighbors = self.neighDict(mIds, scores)
        self.cutNeighborhood(mIds, neighbors)
        matrix = self.buildMatrix(maxMId, maxUId, movieRatings)
        computed = self.neighborhoodRatings(matrix, mIds, neighbors, userIds)
        self.isolateRecc(computed, userIds)

    def evalRatings(self, maxMId, maxUId, mIds, movieRatings, userIds):
        nUniques = np.zeros((maxMId + 1, maxUId + 1))
        for rating in movieRatings:
            nUniques[rating[1]][rating[0]] = rating[2]
        self.normalizeMatrix(nUniques, mIds, userIds)
        initMatrix = dict()
        for ids in mIds:
            initMatrix[ids] = np.linalg.norm(nUniques[ids][:])
        return nUniques, initMatrix

    @property
    def readMovies(self):
        movieRatings = list()
        infile = open(self.file, 'r')
        lines = infile.read().splitlines()
        lines.pop(0)
        for line in lines:
            line = line.split(",")
            colUserID = int(line[0])
            colMovieID = int(line[1])
            colMovieRT = float(line[2])
            movieRatings.append((colUserID, colMovieID, colMovieRT))
        infile.close()
        return movieRatings

    def uniqueColumns(self, movieRatings):
        userIds = set()
        mIds = set()
        for rating in movieRatings:
            userIds.add(rating[0])
            mIds.add(rating[1])
        userIds = sorted(userIds)
        mIds = sorted(mIds)
        maxUId = max(userIds)
        maxMId = max(mIds)
        return maxMId, maxUId, mIds, userIds

    def normalizeMatrix(self, dup, mId, userids):
        for mid in mId:
            total = 0
            for userId in userids:
                if dup[mid][userId] != 0:
                    total += 1
            mean = np.sum(dup[mid][:]) / total
            self.reduceCp(mean, dup, mid, userids)

    def reduceCp(self, avgRating, copy, movieid, userids):
        for userid in userids:
            if copy[movieid][userid] != 0:
                copy[movieid][userid] = copy[movieid][userid] - avgRating

    def compSimilarity(self, dup, mId, mt):
        finalVal = dict()
        for index1 in mId:
            for index2 in mId:
                if index2 > index1:
                    ans = np.sum(dup[index1][:] * dup[index2][:])
                    denom = mt[index1] * mt[index2]
                    if denom != 0:
                        finalVal[(index1, index2)] = ans / denom
                    else:
                        finalVal[(index1, index2)] = -1
        return finalVal

    def neighDict(self, mIds, scores):
        neighbors = dict()
        for index1 in mIds:
            temp = list()
            for index2 in mIds:
                if index1 < index2:
                    temp.append((index2, scores[(index1, index2)]))
                elif index1 > index2:
                    temp.append((index2, scores[(index2, index1)]))
            neighbors[index1] = sorted(temp, key=operator.itemgetter(1), reverse=True)
        return neighbors

    def cutNeighborhood(self, mIds, neighbors):
        for mid in mIds:
            movieT = list()
            argT = list()
            temp = neighbors[mid]
            if len(temp) > self.neighborhood:
                movieT = self.handleT(temp, argT, movieT)
            else:
                movieT = temp
            neighbors[mid] = movieT

    def handleT(self, records, argT, tMovies):
        sT = records[self.neighborhood - 1][1]
        for x in records:
            if x[1] > sT:
                tMovies.append(x)
            elif x[1] == sT:
                argT.append(x)
            else:
                break
        argT.sort(key=operator.itemgetter(0))
        topU = tMovies + argT[0:(self.neighborhood - len(tMovies))]
        return topU

    def buildMatrix(self, mMid, mUid, movieRatings):
        matrix = np.zeros((mMid + 1, mUid + 1))
        for r in movieRatings:
            matrix[r[1]][r[0]] = r[2]
        return matrix

    def neighborhoodRatings(self, matrix, Mid, n, uid):
        Dict = dict()
        for Uid in uid:
            self.neighborhoodHelper(Dict, matrix, Mid, n, Uid)
        return Dict

    def neighborhoodHelper(self, Dict, matrix, mid, neighbors, uid):
        temp = list()
        for Mid in mid:
            if matrix[Mid][uid] == 0:
                numerator = 0
                denominator = 0
                for n in neighbors[Mid]:
                    if matrix[n[0]][uid] != 0:
                        numerator += n[1] * matrix[n[0]][uid]
                        denominator += n[1]
                if denominator > 0:
                    temp.append((Mid, numerator / denominator))
        Dict[uid] = sorted(temp, key=operator.itemgetter(1), reverse=True)

    def isolateRecc(self, Dict, uid):
        finalDict = dict()
        for Uid in uid:
            uT = list()
            argT = list()
            temp = Dict[Uid]
            if len(temp) > self.userTop:
                threshold = temp[self.userTop - 1][1]
                self.breakT(temp, threshold, argT, uT)
                argT.sort(key=operator.itemgetter(0))
                uT = uT + argT[0:(self.userTop - len(uT))]
            else:
                threshold = temp[len(temp) - 1][1]
                self.tiedRatings(temp, threshold, argT, uT)
                argT.sort(key=operator.itemgetter(0))
                uT = uT + argT
            finalDict[Uid] = uT
        self.writeResults(self.outfile, finalDict)

    def tiedRatings(self, temp, cap, argT, uT):
        for r in temp:
            if r[1] > cap:
                uT.append(r)
            else:
                argT.append(r)

    def breakT(self, temp, cap, argT, uT):
        for rating in temp:
            if rating[1] > cap:
                uT.append(rating)
            elif rating[1] == cap:
                argT.append(rating)
            else:
                break

    def writeResults(self, outfile, records):
        file = open(self.outfile, 'w')
        for key, data in sorted(records.items()):
            file.write(str(key))
            for index in data:
                file.write(' ' + "{0}".format(str(index[0])))
            file.write('\n')
        file.close()


def main():
    print("The item-item collab filtering has begun.")
    iicf = IICFilter("movie-lens-data\\ratings.csv")
    print("Now analyzing the movies.")
    iicf.run()
    print("\nThe program is complete. View results in '{0}'.".format(iicf.outfile))


if __name__ == '__main__':
    main()