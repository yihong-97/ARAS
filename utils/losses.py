import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ARCS_loss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, device = 'cuda'):
        super(ARCS_loss, self).__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.first_run = True
        self.device = device

        self.mixed_centroids = None
        self.mixed_centroids_avg = None

        self.source_feat = None
        self.source_feat_mask = None
        self.source_argmax = None
        self.source_argmax_mask = None
        self.source_argmax_full = None
        self.source_softmax = None
        self.source_softmax_mask = None
        self.source_softmax_full = None

        self.target_feat = None
        self.target_feat_mask = None
        self.target_argmax = None
        self.target_argmax_mask = None
        self.target_argmax_full = None
        self.target_softmax = None
        self.target_softmax_mask = None
        self.target_softmax_full = None

        self.source_num_pixel = None
        self.target_num_pixel = None

        self.dist_func = None

        self.B = None
        self.Hs, self.Ws, self.hs, self.ws = None, None, None, None
        self.Ht, self.Wt, self.ht, self.wt = None, None, None, None

    def dist(self, tensor):
        if isinstance(self.dist_func, int):
            return torch.norm(tensor, p=self.dist_func, dim=1)/tensor.size(1)
        else:
            return self.dist_func(tensor)

    def feature_processing_mask(self, feat, softmax, confidence, domain, argmax_dws_type='bilinear'):
        self.B = softmax.size(0)
        if domain == 'source':
            self.Hs, self.Ws, self.hs, self.ws = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)
        else:
            self.Ht, self.Wt, self.ht, self.wt = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)

        feat = feat.permute(0, 2, 3, 1).contiguous()  # size B x h x w x F
        h, w = feat.size(-3), feat.size(-2)
        feat = feat.view(-1, feat.size()[-1])  # size N x F

        peak_values, argmax = torch.max(softmax, dim=1)  # size B x H x W
        if argmax_dws_type == 'nearest': argmax_dws = torch.squeeze(F.interpolate(torch.unsqueeze(argmax.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w
        softmax_dws = F.interpolate(softmax, size=(h, w), mode='bilinear', align_corners=True)  # size B x C x h x w

        if argmax_dws_type == 'bilinear': _, argmax_dws = torch.max(softmax_dws, dim=1)  # size B x h x w


        mask = confidence

        mask = mask.view(-1)


        num_pixel = mask.shape[0]

        # mask_rate = torch.sum(mask).float() / num_pixel
        if domain == 'source':
            mask_rate = torch.sum(mask).float() / num_pixel
        else:
            mask_rate = torch.sum(1 - mask).float() / num_pixel

        argmax_dws = argmax_dws.view(-1)  # size N

        softmax_dws = softmax_dws.permute(0, 2, 3, 1).contiguous()  # size B x h x w x C
        softmax_dws = softmax_dws.view(-1, softmax_dws.size()[-1])  # size N x C

        if domain == 'source':
            self.source_feat = feat # N x F
            self.source_feat_mask = feat[mask,:]  # N‘ x F
            self.source_feat_mask_confidence = mask
            self.source_argmax = argmax_dws  # N
            self.source_argmax_mask = argmax_dws[mask]  # N’
            self.source_argmax_full = torch.max(softmax,dim=1)[1]  # B x H x W
            self.source_softmax = softmax_dws  # N x C
            self.source_softmax_mask = softmax_dws[mask,:]  # N‘ x C
            self.source_softmax_full = softmax  # B x C x H x W
            self.source_num_pixel = mask_rate
        else:
            self.target_feat = feat # N x F
            self.target_feat_confidence = mask  # N’ x F
            self.target_argmax = argmax_dws  # N
            self.target_argmax_full = torch.max(softmax,dim=1)[1]  # B x H x W
            self.target_softmax = softmax_dws  # N x C
            self.target_softmax_full = softmax  # B x C x H x W
            self.target_num_pixel = mask_rate

    def compute_centroids_mixed(self, centroids_smoothing=-1):

        feat_list_source, feat_list_target = [], []
        ns_check, nt_check = 0, 0
        indices = [i for i in range(self.num_class)]
        centroid_list = []

        for i in range(self.num_class):
            source_mask = torch.eq(self.source_argmax_mask.detach(), i)  # size Ns‘
            target_mask = torch.eq(self.target_argmax.detach(), i)  # size Nt’
            # select only features of class i
            source_feat_i = self.source_feat_mask[source_mask, :] # size Ns_i‘ x F
            target_feat_i = self.target_feat[target_mask, :] # size Nt_i’ x F
            target_feat_mask_i = 1 - self.target_feat_confidence[target_mask].view(-1)  # size Nt_i’ x F
            # check if there is at least one feature for source and target class i sets, otherwise insert None in the respective list
            if source_feat_i.size(0) > 0:
                feat_list_source.append(source_feat_i)
                ns_check += source_feat_i.size(0)
            else:
                feat_list_source.append(None)
            if target_feat_i.size(0) > 0:
                feat_list_target.append(target_feat_i)
                nt_check += target_feat_i.size(0)
            else:
                feat_list_target.append(None)
            # compute centroid mean and save it only if class i has at least one feature associated to it, otherwise keep a tensor of python 'Inf' values
            if source_feat_i.size(0) > 0 or target_feat_i.size(0) > 0:
                feat_weight = torch.cat((target_feat_mask_i, torch.ones_like(source_feat_i[:,1])),dim=0).view(1, -1)
                feat_i = torch.mm(feat_weight, torch.cat((target_feat_i, source_feat_i), 0))
                feat_i_num = torch.sum(source_mask) + torch.sum(target_feat_mask_i)
                centroid = torch.div(feat_i, feat_i_num)  # size 1 x F
                centroid_list.append(centroid)
            else:
                centroid_list.append(torch.tensor([[float("Inf")] * self.source_feat_mask.size(1)], dtype=torch.float).to(self.device))  # size 1 x F
                indices.remove(i)

        self.mixed_centroids = torch.squeeze(torch.stack(centroid_list, dim=0))  # size C x 1 x F -> C x F

        if centroids_smoothing >= 0.:
            if self.mixed_centroids_avg is None: self.mixed_centroids_avg = self.mixed_centroids
            self.mixed_centroids_avg = torch.where(self.mixed_centroids_avg != float('inf'), self.mixed_centroids_avg, self.mixed_centroids)
            self.mixed_centroids = torch.where(self.mixed_centroids == float('inf'), self.mixed_centroids_avg.detach(), self.mixed_centroids)
            self.mixed_centroids = centroids_smoothing*self.mixed_centroids + (1-centroids_smoothing)*self.mixed_centroids_avg.detach()
            self.mixed_centroids_avg = self.mixed_centroids.detach().clone()

    def feat_to_centroid(self, feat_domain):

        if feat_domain == 'source':
            feat = self.source_feat
            argmax = self.source_argmax.detach()
        elif feat_domain == 'target':
            feat = self.target_feat
            argmax = self.target_argmax.detach()
        else:
            raise ValueError('Wrong param used: {}    Select from: [source, target]'.format(feat_domain))

        centroids = self.mixed_centroids

        assert feat is not None and centroids is not None

        count, f_dist = 0, 0
        cen_indices = [i for i in range(self.num_class) if (centroids[i, 0] == float('Inf')).item() == 0]
        dist_but_list = []
        dist_i_but_list_max = []
        for i in range(self.num_class):

            if (centroids[i, 0] == float('Inf')).item() == 1: continue

            mask = torch.eq(argmax.detach(), i)  # size N
            feat_i = feat[mask, :]  # size N_i x F

            if feat_i.size(0) == 0: continue


            f_dist = f_dist + torch.mean(self.dist(centroids[i, :] - feat_i))

            count += 1

            indices_but_i = np.array([ind for ind in cen_indices if ind != i])

            if len(indices_but_i) == 0: continue

            dist_i_but_list = []
            for i_inter in indices_but_i:
                dist_i_but = self.dist(centroids[i_inter, :] - feat_i)
                dist_i_but_list.append(dist_i_but)
            dist_i_but_list = torch.stack(dist_i_but_list, dim=1)
            dist_i_but_list_max.append(torch.max(torch.mean(dist_i_but_list, dim=0)))
            dist_but_list.append(torch.mean(dist_i_but_list, dim=0))
        dist_i_but_list_max = torch.stack(dist_i_but_list_max, dim=0)
        dist_i_but_list_max_min = torch.min(dist_i_but_list_max).item()
        dist_but_list = torch.stack(dist_but_list, dim=0)
        dist_but_list[dist_but_list >= dist_i_but_list_max_min] = 0.
        f_dist_inter = torch.mean(dist_but_list)

        return f_dist / count, f_dist_inter

    def intra_domain_c2c(self):

        centroids = self.mixed_centroids

        c_dist = 0
        indices = [i for i in range(self.num_class) if (centroids[i, 0] == float('Inf')).item() == 0]
        for i in indices:
            indices_but_i = np.array([ind for ind in indices if ind != i])
            c_dist = c_dist + torch.mean(self.dist(centroids[i, :] - centroids[indices_but_i, :]))

        return c_dist / len(indices)


    def similarity_dsb(self, feat_domain, temperature=1.):

        if feat_domain == 'source':
            feat = self.source_feat_mask  # size N‘ x F
        elif feat_domain == 'target':
            feat = self.target_feat  # size N’ x F
        elif feat_domain == 'both':
            feat = torch.cat([self.source_feat_mask, self.target_feat], dim=0)  # (Ns‘ + Nt’) x F
        else:
            raise ValueError('Wrong param used: {}    Select from: [source, target, both]'.format(feat_domain))

        centroids = self.mixed_centroids

        # remove centroids of not seen classes
        seen_classes = [i for i in range(self.num_class) if not torch.isnan(centroids[i, 0]) and not centroids[i, 0] == float('Inf')]  # list of C elems, True for seen classes, False elsewhere
        centroids_filtered = centroids[seen_classes, :]  # C_seen x F

        feat_mask_t = 1 - self.target_feat_confidence.detach()
        feat_weight = torch.cat((torch.ones_like(self.source_feat_mask[:, 1]), feat_mask_t), dim=0).view(-1)
        z = torch.mm(feat, centroids_filtered.t())  # size N‘ x C_seen
        z_entropy = torch.sum(F.softmax(z / temperature, dim=1) * F.log_softmax(z / temperature, dim=1), dim=1)
        z_entropy_w = feat_weight * z_entropy
        loss = -1 * torch.mean(z_entropy_w)
        return z, loss

    def clustering_loss(self, clustering_params):

        norm_order = clustering_params['norm_order']
        self.dist_func = norm_order

        f_dist_source, f_inter_dist_source = self.feat_to_centroid(feat_domain='source')
        f_dist_target, f_inter_dist_target = self.feat_to_centroid(feat_domain='target')
        c_dist = self.intra_domain_c2c()

        weight_source = torch.sum(self.source_feat_mask_confidence)
        weight_target = torch.sum(1-self.target_feat_confidence.detach())
        rate_source = weight_source / (weight_source + weight_target)
        rate_target = weight_target / (weight_source + weight_target)
        sep_dis = (c_dist + f_inter_dist_source * rate_target + f_inter_dist_target * rate_source)
        return sep_dis, f_dist_source, f_dist_target

    def entropy_loss(self, entropy_params):

        temp = entropy_params['temp']

        _, loss = self.similarity_dsb(feat_domain='both', temperature=temp)

        return loss

    def forward(self, **kwargs):


        self.feature_processing_mask(feat=kwargs.get('source_feat'), softmax=kwargs.get('source_prob'), confidence=kwargs.get('source_mask'), domain='source')

        self.feature_processing_mask(feat=kwargs.get('target_feat'), softmax=kwargs.get('target_prob'), confidence=kwargs.get('target_mask'), domain='target')


        smo_coeff = kwargs['smo_coeff']
        assert smo_coeff <= 1., 'Centroid smoothing coefficient with invalid value: {}'.format(smo_coeff)
        self.compute_centroids_mixed(centroids_smoothing=smo_coeff)

        sep_dis, f_dist_source, f_dist_target, ent_loss = None, None, None, None

        if 'clustering_params' in kwargs.keys():
            clustering_params = kwargs.get('clustering_params')
            if self.source_num_pixel==0. or self.target_num_pixel == 0.:
                sep_dis, f_dist_source, f_dist_target = torch.tensor(0.).cuda(), torch.tensor(0.).cuda(), torch.tensor(0.).cuda()
            else:
                sep_dis, f_dist_source, f_dist_target = self.clustering_loss(clustering_params)

        torch.cuda.empty_cache()
        if 'entropy_params' in kwargs.keys():
            entropy_params = kwargs.get('entropy_params')
            ent_loss = self.entropy_loss(entropy_params)

        output = {'sep_dis':sep_dis, 'f_dist_source':f_dist_source, 'f_dist_target':f_dist_target, 'ent_loss':ent_loss}
        return output

class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_index)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=self.num_class + 1, min=-1,
                               max=self.num_class - 1).float()
            hist = hist[1:]
            weight = \
            (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(
                argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss


