import _pickle as cPickle
import numpy as np
import torch as t
import torch.nn as nn
from torch import LongTensor as LT
from torch import FloatTensor as FT
import torch.nn.functional as F
from torch.utils.data import Dataset
import pickle

class hmt_grn(nn.Module):

    def __init__(self, arg):
        super(hmt_grn, self).__init__()
        self.hidden_dim = arg['hidden_dim']

        self.embedding_dim = arg['embedding_dim']
        self.num_classes = arg['num_classes']
        self.maxLength = arg['max_length']
        self.padding_idx = 0
        self.dropout = nn.Dropout(p=arg['dropout'])

        self.temporalGAT = multiHeadAttention(arg, 'poi').float()
        self.spatialGAT = multiHeadAttention(arg, 'poi').float()

        self.poiEmbed = nn.Embedding(arg['vocab_size'], arg['embedding_dim'], padding_idx=self.padding_idx)
        self.geoHashEmbed2 = nn.Embedding(len(arg['index2geoHash_2']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed3 = nn.Embedding(len(arg['index2geoHash_3']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed4 = nn.Embedding(len(arg['index2geoHash_4']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed5 = nn.Embedding(len(arg['index2geoHash_5']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.geoHashEmbed6 = nn.Embedding(len(arg['index2geoHash_6']), arg['embedding_dim'],
                                          padding_idx=self.padding_idx)
        self.userEmbed = nn.Embedding(arg['numUsers'], arg['userEmbed_dim'], padding_idx=self.padding_idx)

        self.nextGeoHash2Dense = nn.Linear(arg['hidden_dim'] + arg['embedding_dim'], len(arg['geohash2Index_2']),
                                           bias=True)

        self.nextGeoHash3Dense = nn.Linear(arg['hidden_dim'] + arg['embedding_dim'], len(arg['geohash2Index_3']),
                                           bias=True)

        self.nextGeoHash4Dense = nn.Linear(arg['hidden_dim'] + arg['embedding_dim'], len(arg['geohash2Index_4']),
                                           bias=True)

        self.nextGeoHash5Dense = nn.Linear(arg['hidden_dim'] + arg['embedding_dim'], len(arg['geohash2Index_5']),
                                           bias=True)

        self.nextGeoHash6Dense = nn.Linear(arg['hidden_dim'] + arg['embedding_dim'], len(arg['geohash2Index_6']),
                                           bias=True)

        self.fuseDense = nn.Linear(arg['userEmbed_dim'] + arg['hidden_dim'],
                                   arg['num_classes'] * 1, bias=True)

        self.ownLSTM = ownLSTM(arg['embedding_dim'] * 1, arg['hidden_dim'])

    def forward(self, input, mode, arg):


        if mode == 'train':
            x, users, y, x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 = input
            x, users, y = LT(x).cuda(), LT(users).cuda(), LT(y).cuda()

            numTimeSteps = len(x[0])

        else:

            x, users, x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 = input
            x, users, x_geoHash2, x_geoHash3, x_geoHash4, x_geoHash5, x_geoHash6 = LT(x).cuda(), LT(users).cuda(), LT(
            x_geoHash2).cuda(), LT(x_geoHash3).cuda(), \
            LT(x_geoHash4).cuda(), LT(x_geoHash5).cuda(), LT(x_geoHash6).cuda()

            users = users.unsqueeze(0)
            numTimeSteps = len(x)

        batchSize = len(x)

        x_embed_allTimeStep = self.poiEmbed(x)
        x_embed_allTimeStep = x_embed_allTimeStep.view(batchSize, numTimeSteps,self.embedding_dim)

        finalAttendedOutSpatial = FT([]).cuda()
        finalAttendedOutTemporal = FT([]).cuda()

        # ================================ST Graphs============================
        for eachBatchIndex in range(batchSize):

            currentSample = x[eachBatchIndex]
            if mode == 'test':
                currentSample = currentSample.unsqueeze(0)
            singleBatchSpatial = FT([]).cuda()
            singleBatchTemporal = FT([]).cuda()
            for timeStep in range(len(currentSample)):
                sample = currentSample[timeStep].item()

                spatialNiegh = [i for i in arg['spatialGraph'][sample]]
                try:
                    temporalNiegh = [i for i in arg['temporalGraph'][sample]]
                except:
                    # as the temporal graph is based only on training set, there might be new POIs not found in the graph.
                    # in that case, we use only the POI itself as neighbour.
                    temporalNiegh = [sample]

                x_embed = self.poiEmbed(LT([sample]).cuda())

                if sample == 0: # padding
                    spatial_GAT_out = x_embed
                    temporal_GAT_out = x_embed
                else:
                    spatialNieghEmbed = self.poiEmbed(LT(spatialNiegh).cuda()).unsqueeze(0)
                    temporalNieghEmbed = self.poiEmbed(LT(temporalNiegh).cuda()).unsqueeze(0)

                    spatial_GAT_out = self.spatialGAT(x_embed, spatialNieghEmbed, numTimeSteps, users, 'repeat', arg)
                    temporal_GAT_out = self.temporalGAT(x_embed, temporalNieghEmbed, numTimeSteps, users, 'repeat', arg)

                singleBatchSpatial = t.cat((singleBatchSpatial, spatial_GAT_out), dim=0)
                singleBatchTemporal = t.cat((singleBatchTemporal, temporal_GAT_out), dim=0)

            finalAttendedOutSpatial = t.cat((finalAttendedOutSpatial, singleBatchSpatial.unsqueeze(0)), dim=0)
            finalAttendedOutTemporal = t.cat((finalAttendedOutTemporal, singleBatchTemporal.unsqueeze(0)), dim=0)
        # ================================ST Graphs============================

        x_geoHash_embed2 = self.dropout(self.geoHashEmbed2(x_geoHash2))
        x_geoHash_embed3 = self.dropout(self.geoHashEmbed3(x_geoHash3))
        x_geoHash_embed4 = self.dropout(self.geoHashEmbed4(x_geoHash4))
        x_geoHash_embed5 = self.dropout(self.geoHashEmbed5(x_geoHash5))
        x_geoHash_embed6 = self.dropout(self.geoHashEmbed6(x_geoHash6))

        rnn_out, _ = self.ownLSTM(x_embed_allTimeStep, finalAttendedOutSpatial, finalAttendedOutTemporal)

        userEmbedSeq = self.userEmbed(users)

        userEmbedSeq = self.dropout(userEmbedSeq)
        rnn_out = self.dropout(rnn_out)

        finalEmbed = t.cat((rnn_out, userEmbedSeq), dim=2)

        nextGeoHashlogits_2 = self.nextGeoHash2Dense(t.cat((rnn_out, x_geoHash_embed2), dim=2))
        nextGeoHashlogits_3 = self.nextGeoHash3Dense(t.cat((rnn_out, x_geoHash_embed3), dim=2))
        nextGeoHashlogits_4 = self.nextGeoHash4Dense(t.cat((rnn_out, x_geoHash_embed4), dim=2))
        nextGeoHashlogits_5 = self.nextGeoHash5Dense(t.cat((rnn_out, x_geoHash_embed5), dim=2))
        nextGeoHashlogits_6 = self.nextGeoHash6Dense(t.cat((rnn_out, x_geoHash_embed6), dim=2))

        nextGeoHashlogits_2 = nextGeoHashlogits_2.view(batchSize, numTimeSteps, len(arg['geohash2Index_2']))
        nextGeoHashlogits_3 = nextGeoHashlogits_3.view(batchSize, numTimeSteps, len(arg['geohash2Index_3']))
        nextGeoHashlogits_4 = nextGeoHashlogits_4.view(batchSize, numTimeSteps, len(arg['geohash2Index_4']))
        nextGeoHashlogits_5 = nextGeoHashlogits_5.view(batchSize, numTimeSteps, len(arg['geohash2Index_5']))
        nextGeoHashlogits_6 = nextGeoHashlogits_6.view(batchSize, numTimeSteps, len(arg['geohash2Index_6']))

        nextgeohashPred_2 = F.log_softmax(nextGeoHashlogits_2, dim=2)
        nextgeohashPred_3 = F.log_softmax(nextGeoHashlogits_3, dim=2)
        nextgeohashPred_4 = F.log_softmax(nextGeoHashlogits_4, dim=2)
        nextgeohashPred_5 = F.log_softmax(nextGeoHashlogits_5, dim=2)
        nextgeohashPred_6 = F.log_softmax(nextGeoHashlogits_6, dim=2)

        logits = self.fuseDense(finalEmbed)

        logits = logits.view(batchSize, numTimeSteps, self.num_classes)

        if mode == 'train':

            output = F.log_softmax(logits, dim=2)
            return output, nextgeohashPred_2, nextgeohashPred_3, nextgeohashPred_4, nextgeohashPred_5, nextgeohashPred_6


        elif mode == 'test':

            softmaxScores = F.softmax(logits, dim=2)
            nextgeohashPred_2_test = F.softmax(nextGeoHashlogits_2, dim=2)
            nextgeohashPred_3_test = F.softmax(nextGeoHashlogits_3, dim=2)
            nextgeohashPred_4_test = F.softmax(nextGeoHashlogits_4, dim=2)
            nextgeohashPred_5_test = F.softmax(nextGeoHashlogits_5, dim=2)
            nextgeohashPred_6_test = F.softmax(nextGeoHashlogits_6, dim=2)
            return softmaxScores, nextgeohashPred_2_test, nextgeohashPred_3_test, nextgeohashPred_4_test, nextgeohashPred_5_test, nextgeohashPred_6_test

class multiHeadAttention(nn.Module):
    def __init__(self, arg, type):
        super(multiHeadAttention, self).__init__()
        self.numHeads = 1
        self.heads = {}

        for i in range(self.numHeads):
            self.heads[i] = selfAttention(arg, type).cuda().float()

    def forward(self, mainNodeEmbed, neighbourNodesEmbeds, numTimeSteps, users, mode, arg):

        allHeads = FT([]).cuda()

        for i in range(self.numHeads):
            output_x = self.heads[i](mainNodeEmbed, neighbourNodesEmbeds, numTimeSteps, users, mode,arg)

            output_x = output_x.unsqueeze(1)
            allHeads = t.cat((allHeads, output_x), dim=1)

        final_x = allHeads.mean(dim=1)
        return final_x


class selfAttention(nn.Module):
    def __init__(self, arg, type):
        super(selfAttention, self).__init__()
        if type == 'user':
            dim = arg['userEmbed_dim']
        elif type == 'poi':
            dim = arg['embedding_dim']

        self.embedding_dim = dim
        self.attentionDense = nn.Linear((dim * 2), dim, bias=True)

        nn.init.xavier_normal_(self.attentionDense.weight)

        self.w = nn.Linear(dim, dim, bias=False)
        self.padding_idx = 0
        self.leakyRelu = nn.LeakyReLU()

    def forward(self, mainNodeEmbed, neighbourNodesEmbeds, numTimeSteps, users, mode, arg):

        numNeighbours = neighbourNodesEmbeds.shape[1]

        projected_x, projected_neigh = self.w(mainNodeEmbed), self.w(neighbourNodesEmbeds)
        if mode == 'repeat':
            projected_x = projected_x.unsqueeze(1)

            projected_x = projected_x.expand(-1, numNeighbours, -1)


        concat_x_neigh = t.cat((projected_x, projected_neigh), dim=2)
        e_ij = self.attentionDense(concat_x_neigh)
        e_ij = self.leakyRelu(e_ij)
        a_ij = t.softmax(e_ij,dim=1)

        new_x = a_ij * projected_neigh

        output_x = new_x.sum(dim=1)

        return output_x




class classificationDataset(Dataset):

    def __init__(self, numUsers, dataSource, arg):

        if dataSource == 'gowalla':
            data_root = 'data/' + arg['dataFolder'] + '/gowalla_train.pkl'
        elif dataSource == 'global_scale':
            data_root = 'data/' + arg['dataFolder'] + '/global_scale_train.pkl'

        poiFileName = 'data/' + arg['dataFolder'] + '/' + dataSource + '_poiCount.pickle'

        with open(poiFileName,'rb') as handle:
            n_poi = pickle.load(handle)

        arg['vocab_size'] = n_poi
        arg['num_classes'] = n_poi
        arg['numUsers'] = numUsers

        with open(data_root, 'rb') as f:
            pois_seq, delta_t_seq, delta_d_seq = cPickle.load(f)

        arg['user2TrainSet'] = {}
        for userID, sample in enumerate(pois_seq, start=1):
            arg['user2TrainSet'][userID] = sample

        arg['user2TrainLength'] = {}
        for userID, sample in enumerate(pois_seq, start=1):
            arg['user2TrainLength'][userID] = len(sample)

        x_train = np.array([seq[:-1] for seq in pois_seq])
        y_train = np.array([seq[1:] for seq in pois_seq])
        arg['max_length'] = np.max([len(i) for i in x_train])

        assert x_train.shape == y_train.shape
        assert len(x_train)

        with open('data/' + arg['dataFolder'] + '/' + dataSource + '_usersData.pickle',
                  'rb') as handle:
            userData = pickle.load(handle)
        trainUser, testUser = userData
        trainUser = [seq[:-1] for seq in trainUser]

        trainUser = np.array(trainUser)


        # =======================padding for batch only=======================

        def zeroPaddingToRight(data, paddingIndex):
            max_length = arg['max_length']
            paddedOutput = [i + [paddingIndex] * (max_length - len(i)) if
                            len(i) < arg['max_length'] else i for i in data]
            return paddedOutput

        padding_index = 0

        trainUser = np.array(zeroPaddingToRight(trainUser, padding_index))
        x_train = np.array(zeroPaddingToRight(x_train, padding_index))
        y_train = np.array(zeroPaddingToRight(y_train, padding_index))
        # =======================padding for batch only=======================

        self.data = [(x_train[i], trainUser[i], y_train[i]) for i in range(x_train.shape[0])]

        print('Classification Data Loaded')


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        poiSeq, users, truthSeq = self.data[idx]
        return np.array(poiSeq), np.array(users), np.array(truthSeq)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.W = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.s_W = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.t_W = nn.Linear(input_size, 4 * hidden_size, bias=False)

    def forward(self, x, hidden, spatial_per_step, temporal_per_step):
        if hidden is None:
            hidden = self._init_hidden(x)
        h_t, c_t = hidden

        previous_h_t = h_t
        previous_c_t = c_t
        x = x

        allGates_preact = self.W(x) + self.U(previous_h_t) + self.s_W(spatial_per_step) + self.t_W(
            temporal_per_step)

        input_g_ceilingIndex = self.hidden_size
        forget_g_ceilingIndex = 2 * self.hidden_size
        output_g_ceilingIndex = 3 * self.hidden_size
        cell_g_ceilingIndex = 4 * self.hidden_size

        input_g = allGates_preact[:, :input_g_ceilingIndex].sigmoid()
        forget_g = allGates_preact[:, input_g_ceilingIndex:forget_g_ceilingIndex].sigmoid()
        output_g = allGates_preact[:, forget_g_ceilingIndex:output_g_ceilingIndex].sigmoid()
        c_t_g = allGates_preact[:, output_g_ceilingIndex:cell_g_ceilingIndex].tanh()

        c_t = forget_g * previous_c_t + input_g * c_t_g
        h_t = output_g * c_t.tanh()

        batchSize = x.shape[0]
        h_t = h_t.view(batchSize, self.hidden_size)
        c_t = c_t.view(batchSize, self.hidden_size)

        return h_t, c_t

    def _init_hidden(self, x):

        batchSize = x.shape[0]

        h = t.zeros([batchSize, self.hidden_size]).cuda()
        c = t.zeros([batchSize, self.hidden_size]).cuda()
        return h, c


class ownLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.hidden_size = hidden_size

    def forward(self, input_, spatial, temporal, hidden=None):


        input_ = t.transpose(input_, 1, 0)
        batchSize = input_.shape[1]

        spatial = t.transpose(spatial, 1, 0)
        temporal = t.transpose(temporal, 1, 0)

        outputs = FT([]).cuda()

        for x, spatial_per_step, temporal_per_step in zip(t.unbind(input_, dim=0), t.unbind(spatial, dim=0),
                                                          t.unbind(temporal, dim=0)):

            hidden = self.lstm_cell(x, hidden, spatial_per_step, temporal_per_step)

            h_t, c_t = hidden

            outputs = t.cat((outputs, h_t.clone().view(1, batchSize, self.hidden_size)), dim=0)

        outputs = outputs.permute(1, 0, 2)
        return outputs, None


