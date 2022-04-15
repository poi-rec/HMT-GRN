# -*- coding: utf-8 -*-
import os
import sys
from utils.func import *
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import time
import networkx as nx
from model import *
import random

if __name__ == '__main__': 

    for dataSource in ['gowalla']:
        arg = {}

        start_time = time.time()

        arg['epoch'] = 20
        arg['beamSize'] = 100
        arg['embedding_dim'] = 1024
        arg['userEmbed_dim'] = 1024
        arg['hidden_dim']= 1024
        arg['classification_learning_rate'] = 0.0001
        arg['classification_batch'] = 32
        arg['dropout'] = 0.9
        arg['dataFolder'] = 'processedFiles'

        print()
        print(dataSource)
        print()
        print(arg)

        # ==================================Spatial Temporal graphs================================================
        arg['temporalGraph'] =  nx.read_edgelist('data/' + arg['dataFolder'] + '/' + dataSource + '_temporal.edgelist', nodetype=int,
                             create_using=nx.Graph())

        arg['spatialGraph'] = nx.read_edgelist(
            'data/' + arg['dataFolder'] + '/' + dataSource + '_spatial.edgelist', nodetype=int,
            create_using=nx.Graph())
        # ==================================Spatial Temporal graphs================================================

        userFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_userCount.pickle'

        with open(userFileName,
                  'rb') as handle:
            arg['numUser'] = pickle.load(handle)

        print('Data loading')

        # ==================================geohash related data================================================
        for eachGeoHashPrecision in [6,5,4,3,2]:
            poi2geohashFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poi2geohash' + '_' + str(
                eachGeoHashPrecision)
            geohash2poiFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2poi' + '_' + str(
                eachGeoHashPrecision)
            geohash2IndexFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_geohash2Index' + '_' + str(
                eachGeoHashPrecision)

            with open(poi2geohashFileName + '.pickle', 'rb') as handle:
                arg['poi2geohash'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2poiFileName + '.pickle', 'rb') as handle:
                arg['geohash2poi'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)
            with open(geohash2IndexFileName + '.pickle', 'rb') as handle:
                arg['geohash2Index'+'_'+str(eachGeoHashPrecision)] = pickle.load(handle)

            arg['index2geoHash'+'_'+str(eachGeoHashPrecision)] = {v: k for k, v in arg['geohash2Index'+'_'+str(eachGeoHashPrecision)].items()}

        beamSearchHashDictFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_beamSearchHashDict'
        with open(beamSearchHashDictFileName + '.pickle', 'rb') as handle:
            arg['beamSearchHashDict'] = pickle.load(handle)
        # ==================================geohash related data================================================

        classification_dataset = classificationDataset(arg['numUser'], dataSource, arg)

        classification_dataloader = DataLoader(classification_dataset, batch_size=arg['classification_batch'],
                                               shuffle=True, pin_memory=True,
                                               num_workers=0)
        print('Data loaded')
        print('init model')

        classification = hmt_grn(arg).float().cuda()

        classification_optim = Adam(classification.parameters(), lr=arg['classification_learning_rate'])

        print('init model done')

        criterion = nn.NLLLoss(reduction='mean', ignore_index=0)

        nextGeoHashCriterion_2 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_3 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_4 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_5 = nn.NLLLoss(reduction='mean', ignore_index=0)
        nextGeoHashCriterion_6 = nn.NLLLoss(reduction='mean', ignore_index=0)

        for epoch in range(1, arg['epoch'] + 1):

            avgLossDict = {}

            print()
            print('Epoch: ' + str(epoch))

            avgLossDict['Next POI Classification'] = []

            classification_pbar = tqdm(classification_dataloader)

            classification_pbar.set_description('[' + dataSource + "_Classification-Epoch {}]".format(epoch))

            for x, user, y in classification_pbar:

                actualBatchSize = x.shape[0]

                batchLoss = 0

                x_geoHash2 = LT([]).cuda()
                x_geoHash3 = LT([]).cuda()
                x_geoHash4 = LT([]).cuda()
                x_geoHash5 = LT([]).cuda()
                x_geoHash6 = LT([]).cuda()
                for eachBatch in range(x.shape[0]):
                    sample = x[eachBatch].tolist()

                    mappedGeohash = [ arg['geohash2Index'+'_2'][arg['poi2geohash'+'_2'][i]] for i in sample]
                    x_geoHash2 = t.cat((x_geoHash2,LT(mappedGeohash).unsqueeze(0).cuda()),dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_3'][arg['poi2geohash' + '_3'][i]] for i in sample]
                    x_geoHash3 = t.cat((x_geoHash3, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_4'][arg['poi2geohash' + '_4'][i]] for i in sample]
                    x_geoHash4 = t.cat((x_geoHash4, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_5'][arg['poi2geohash' + '_5'][i]] for i in sample]
                    x_geoHash5 = t.cat((x_geoHash5, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)

                    mappedGeohash = [arg['geohash2Index' + '_6'][arg['poi2geohash' + '_6'][i]] for i in sample]
                    x_geoHash6 = t.cat((x_geoHash6, LT(mappedGeohash).unsqueeze(0).cuda()), dim=0)


                input = (x, user, y, x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6)

                logSoftmaxScores,nextgeohashPred_2,nextgeohashPred_3,nextgeohashPred_4,nextgeohashPred_5,nextgeohashPred_6= classification(input, 'train', arg)

                truth = LT(y).cuda()

                #map truth to geohash
                truthDict={}
                for eachGeoHashPrecision in [6,5, 4, 3, 2]:
                    name = 'nextGeoHashTruth'+'_'+ str(eachGeoHashPrecision)
                    behind = '_' + str(eachGeoHashPrecision)
                    truthDict[name] = LT([]).cuda()
                    for eachBatch in range(truth.shape[0]):
                        sample = truth[eachBatch].tolist()
                        mappedNextGeohashTruth = [ arg['geohash2Index'+behind][arg['poi2geohash'+behind][i]] for i in sample]
                        truthDict[name] = t.cat((truthDict[name],LT(mappedNextGeohashTruth).unsqueeze(0).cuda()),dim=0)


                class_size = logSoftmaxScores.shape[2]

                classification_loss = criterion(logSoftmaxScores.view(-1, class_size), truth.view(-1))
                nextGeoHash_loss_2 = nextGeoHashCriterion_2(nextgeohashPred_2.view(-1, len(arg['geohash2Index_2'])), truthDict['nextGeoHashTruth_2'].view(-1))
                nextGeoHash_loss_3 = nextGeoHashCriterion_3(nextgeohashPred_3.view(-1, len(arg['geohash2Index_3'])), truthDict['nextGeoHashTruth_3'].view(-1))
                nextGeoHash_loss_4 = nextGeoHashCriterion_4(nextgeohashPred_4.view(-1, len(arg['geohash2Index_4'])), truthDict['nextGeoHashTruth_4'].view(-1))
                nextGeoHash_loss_5 = nextGeoHashCriterion_5(nextgeohashPred_5.view(-1, len(arg['geohash2Index_5'])), truthDict['nextGeoHashTruth_5'].view(-1))
                nextGeoHash_loss_6 = nextGeoHashCriterion_6(nextgeohashPred_6.view(-1, len(arg['geohash2Index_6'])), truthDict['nextGeoHashTruth_6'].view(-1))

                batchLoss = (classification_loss  +  nextGeoHash_loss_2 +  nextGeoHash_loss_3 + nextGeoHash_loss_4 + nextGeoHash_loss_5 + nextGeoHash_loss_6)  / 6 / actualBatchSize

                classification_optim.zero_grad()
                batchLoss.backward(retain_graph=False)
                classification_optim.step()
                classification_pbar.set_postfix(loss=classification_loss.item()/actualBatchSize)

                avgLossDict['Next POI Classification'].append(classification_loss.item()/actualBatchSize)


            avgLossDict['Next POI Classification'] = np.average(avgLossDict['Next POI Classification'])

            print('Next POI Classification Avg Loss: ' + str(avgLossDict['Next POI Classification']))

            print()

            sys.stdout.flush()

            print()
            print("----- END SAVING  : %s seconds ---" % (time.time() - start_time))
            print()


            if epoch % 20 == 0:

                print()
                print('Epoch ' + str(epoch) + ' Evaluation Start!')
                print()
                model = classification

                arg['novelEval'] = False
                evaluate(model, dataSource, arg)

                arg['novelEval'] = True
                evaluate(model, dataSource, arg)

            sys.stdout.flush()
