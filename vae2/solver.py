"""solver.py"""

import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

if os.getcwd().split('/')[-1] == 'vae2':
    from utils import cuda, grid2gif
    from model import BetaVAE_H, BetaVAE_B, BetaVAE_C, reparametrize
    from dataset import return_data, get_image
else:
    from .utils import cuda, grid2gif
    from .model import BetaVAE_H, BetaVAE_B, BetaVAE_C, reparametrize
    from .dataset import return_data, get_image

import numpy as np
import random
from ray import tune

import pickle
import cv2

SMOOTHING_FACTOR = 0.99

#ray_is_running = tune.session._session is not None

vis = visdom.Visdom()

opts = {
  'layoutopts': {
    'plotly': {
      'yaxis': {
        'type': 'log',
        #'range': [-10, 2],
        #'autorange': False,
      }
    }
  }
}
#vis.line([0], [0], win='recon_loss', opts=opts)

PLOT_WINS_NOT_USED = [
        'flattening_loss',
        'grid_error_plot',
        ]

PLOT_WINS = [
        'recon_loss_all',
        'recon_loss_smoothed',
        'loss',
        'logvar',
        'outside_boundary_weight',
        'euclid_metric',
        'euclid_metric_smoothed',
        ]

for win_name in PLOT_WINS:
    vis.line([0], [0], win=win_name, opts={**opts, **dict(title=win_name)})


#vis.line([0], [0], win='recon_loss_all', opts={**opts, **dict(title='recon_loss_all')})
#vis.line([0], [0], win='recon_loss_smoothed', opts=opts)
#vis.line([0], [0], win='grid_error_plot', opts=opts)
#vis.line([0], [0], win='loss', opts=opts)
#vis.line([0], [0], win='logvar', opts=opts)
#vis.line([0], [0], win='outside_boundary_weight', opts=opts)
#vis.line([0], [0], win='flattening_loss')
#vis.line([0, 0], [0, 0], win='loss_stacked', 
#        opts=dict(
#            fillarea=True,
#            showlegend=False,
#            width=800,
#            height=800,
#            xlabel='Time',
#            ylabel='Volume',
#            ytype='log',
#            title='Stacked area plot',
#            marginleft=30,
#            marginright=30,
#            marginbottom=80,
#            margintop=30,
#        ))
#vis.close('euclid_metric')


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def kl_divergence_parameterized(mu, logvar, mu_weight=1, var_weight=1):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + (var_weight * (logvar - logvar.exp()))  - (mu_weight * mu.pow(2)) )
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        print("args:", args)
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.ray = args.ray
        self.args = args

        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'goal_pos':
            self.nc = 1
            #self.decoder_dist = 'bernoulli'
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == 'goal_pos_pre':
            self.nc = 1
            #self.decoder_dist = 'bernoulli'
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == 'ihgg':
            self.nc = 1
            #self.decoder_dist = 'bernoulli'
            self.decoder_dist = 'bernoulli'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        elif args.model == 'C':
            net = BetaVAE_C
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            # self.viz = visdom.Visdom(port=self.viz_port)
            self.viz = vis

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()
        self.grid_error_plot_item = 0

    def get_pos_encoding(self, e, encoding_map_res):
        if type(e) == list:
            e_list = e
        else:
            e_list = [e]

        return_list = []
        for ee in e_list:
            return_list.append((ee + 1) * int(encoding_map_res / 2))

        if type(e) == list:
            return return_list
        else:
            return return_list[0]


    def get_xy_encodings(self, mu, encoding_map_res):
        xys = []
        for e in mu:
            position = [self.get_pos_encoding(ee, encoding_map_res) for ee in e]
            position = [e for e in position if e > 0 and e < encoding_map_res]
            if len(position) == 2:
                #xe, ye = position
                xys.append(position)

        return xys




    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        #if not self.ray:

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        batch_num = 0
        loss_smoothed = 0
        outside_boundary_weight = 1
        flattening_loss_smoothed = 0
        mean_diff_adapted_smoothed = 0
        flattening_loss = 0
        encoding_mus_map_diff_mean = None
        ENCODING_MAP_RES = 10
        encoding_mus_map = np.zeros_like(np.tile(np.arange(1, ENCODING_MAP_RES + 1), (ENCODING_MAP_RES, 1)), dtype=np.float64)
        while not out:
            for x in self.data_loader:
                batch_num += 1
                self.global_iter += 1
                #if not ray_is_running:
                if not self.ray:
                    print("pbar:", pbar)
                    pbar.update(1)
                else:
                    if self.global_iter % 100 == 0:
                        print("self.global_iter:", self.global_iter)


                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                #vis.line([logvar.cpu().detach().numpy()[0]], [batch_num], win='logvar')
                if self.objective == 'C':
                    #mu = mu.clip(-1, 1)
                    pass

                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                loss_smoothed = SMOOTHING_FACTOR * loss_smoothed + (1 - SMOOTHING_FACTOR) * float(recon_loss)


                if self.objective == 'C':
                    total_kld, dim_wise_kld, mean_kld = kl_divergence_parameterized(mu, logvar, mu_weight=1, var_weight=1)
                else:
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                elif self.objective == 'C':
                    #beta_vae_loss = recon_loss + np.sum(mu)
                    #beta_vae_loss = recon_loss + abs(abs(torch.sum(mu)) - abs(torch.sum(logvar)))
                    #print("abs(mu) > 1:", abs(mu) > 1)
                    #beta_vae_loss = recon_loss
                    beta_vae_loss = recon_loss * 0


                    #if torch.any(abs(mu) > 1):
                    #    beta_vae_loss += 100000

                    #outside_boundary_weight = recon_loss.clone().detach()

                    if True:
                        box_loss = 0
                        all_inside = True
                        for xy in mu:
                            for e in xy:
                                if abs(e) > 1:
                                    box_loss += abs(e) * outside_boundary_weight
                                    all_inside = False
                                    #beta_vae_loss += abs(e)**2 * 1
                                    #beta_vae_loss += abs(e) * outside_boundary_weight
                                    #beta_vae_loss = torch.sum(torch.abs(mu))

                        if all_inside:
                            outside_boundary_weight *= 0.99
                        else:
                            outside_boundary_weight *= 1.01
                    if True:
                        box_loss_2 = 0
                        all_inside = True
                        for xy in mu:
                            for e in xy:
                                if abs(e) > 1:
                                    box_loss_2 += (abs(e) - 1)**2 * outside_boundary_weight
                                    all_inside = False
                                    #beta_vae_loss += abs(e)**2 * 1
                                    #beta_vae_loss += abs(e) * outside_boundary_weight
                                    #beta_vae_loss = torch.sum(torch.abs(mu))

                        if all_inside:
                            outside_boundary_weight *= 0.99
                        else:
                            outside_boundary_weight *= 1.01

                    if False:
                        rand_distances_sum = 0
                        for i in range(int(len(mu)/2)):
                            mu1, mu2 = mu[2*i], mu[2*i + 1] 
                            distance = ((mu1[0] - mu2[0])**2 + (mu1[1] - mu2[1])**2)**0.5
                            rand_distances_sum += distance

                        beta_vae_loss -= rand_distances_sum

                    if False:
                        distance_delta_sum = 0
                        #for i in range(int(len(mu)/2)):
                        exp_delta = 2/64
                        for index in range(2):
                            for mu1 in mu:
                                min_abs_delta = float('inf')
                                for mu2 in mu:
                                    #mu1, mu2 = mu[2*i], mu[2*i + 1] 
                                    abs_delta = abs(mu1 - mu2)[index]
                                    if abs_delta < min_abs_delta:
                                        min_abs_delta = abs_delta
                                distance_delta_sum += abs(abs_delta - exp_delta)
                                #distance = ((mu1[0] - mu2[0])**2 + (mu1[1] - mu2[1])**2)**0.5
                                #rand_distances_sum += distance

                        beta_vae_loss += distance_delta_sum

                    if False:
                        flattening_loss = 0
                        map_mean = np.mean(encoding_mus_map)
                        encoding_mus_map_diff_mean = encoding_mus_map - map_mean
                        for xy in self.get_xy_encodings(mu, ENCODING_MAP_RES):
                            xea, yea = xy 
                            xe, ye = [int(e.cpu().detach()) for e in xy]
                            if False:
                                flattening_loss -= abs(xea) * (encoding_mus_map_diff_mean[xe, 0]/abs(xea).detach())
                                flattening_loss -= abs(yea) * (encoding_mus_map_diff_mean[0, ye]/abs(yea).detach())
                            else:
                                for cor in xy:
                                    flattening_loss -= abs(cor) * (encoding_mus_map_diff_mean[xe, ye]/abs(cor).detach())

                        flattening_loss *= 0
                            

                    if True:
                        flattening_loss_2 = 0
                        map_mean = np.mean(encoding_mus_map)
                        encoding_mus_map_grad = np.gradient(encoding_mus_map)
                        if False:
                            encoding_mus_map_diff_mean = encoding_mus_map - map_mean
                        for xy in self.get_xy_encodings(mu, ENCODING_MAP_RES):
                            xea, yea = xy 
                            xe, ye = [int(e.cpu().detach()) for e in xy]
                            xear, year = [2*((e/ENCODING_MAP_RES)-0.5) for e in xy]
                            #print("xe:", xe)
                            #print("ye:", ye)
                            #print("encoding_mus_map_grad:", encoding_mus_map_grad)
                            #print("encoding_mus_map_grad:", encoding_mus_map_grad.shape)
                            #flattening_loss_2 += encoding_mus_map_grad[0, 0, 0]
                            flattening_loss_2 += xear * encoding_mus_map_grad[0][xe, ye]
                            flattening_loss_2 += year * encoding_mus_map_grad[1][xe, ye]

                        #for xy in enumerate(self.get_xy_encodings(mu, ENCODING_MAP_RES)):

                        #if False:
                        #    for mue in mu:
                        #        print("mue:", mue)
                        #        xy = self.get_xy_encodings([mue], ENCODING_MAP_RES)
                        #        xm, ym = mue
                        #        xea, yea = xy 
                        #        xe, ye = [int(e.cpu().detach()) for e in xy]
                        #        #print("xe:", xe)
                        #        #print("ye:", ye)
                        #        #print("encoding_mus_map_grad:", encoding_mus_map_grad)
                        #        #print("encoding_mus_map_grad:", encoding_mus_map_grad.shape)
                        #        #flattening_loss_2 += encoding_mus_map_grad[0, 0, 0]
                        #        flattening_loss_2 += abs(xea) * encoding_mus_map_grad[0][xe, ye]
                        #        flattening_loss_2 += abs(yea) * encoding_mus_map_grad[1][xe, ye]

                        flattening_loss_2 *= .01




                    #if beta_vae_loss == 0:
                    #    beta_vae_loss = recon_loss

                    #if beta_vae_loss == 0:
                    if False:
                        beta_vae_loss += flattening_loss
                        beta_vae_loss += box_loss
                    if True:
                        beta_vae_loss += flattening_loss_2
                        beta_vae_loss += box_loss_2
                        beta_vae_loss += self.beta * total_kld[0]
                        beta_vae_loss += recon_loss


                    #beta_vae_loss += abs(torch.sum(logvar))

                #for e in mu:
                #    position = [int(ee) for ee in (e + 1) * int(ENCODING_MAP_RES / 2) if ee > 0 and ee < ENCODING_MAP_RES]
                #    if len(position) == 2:
                #        xe, ye = position
                #        encoding_mus_map[xe, ye] += 1
                #        encoding_mus_map *= 0.9999
                #for xe, ye in self.get_xy_encodings(mu, ENCODING_MAP_RES):
                for xy in self.get_xy_encodings(mu, ENCODING_MAP_RES):
                    xe, ye = [int(e.cpu().detach()) for e in xy]
                    encoding_mus_map[xe, ye] += 1
                    #encoding_mus_map *= 0.999999
                    #encoding_mus_map *= 0.99999
                    encoding_mus_map *= 1


                #self.euclidean_grid_eval()
                self.gather.insert(images=x.data)
                self.gather.insert(images=F.sigmoid(x_recon).data)
                self.viz_reconstruction_2()
                self.gather.flush()
                if batch_num % 100 == 0:
                    #vis.image(x_recon, win='x_recon')
                    #self.euclidean_eval()
                    if False:
                        mean_diff_adapted = self.euclidean_grid_eval()
                        #vis.line([mean_diff_adapted], [batch_num], win='euclid_metric', update='append',  opts={**opts, **{'title:': 'euclid_metric'}})
                        vis.line([mean_diff_adapted], [batch_num], win='euclid_metric', update='append')
                        MEAN_DIFF_ADAPTED_SMOOTHING_FACTOR = 0.9
                        mean_diff_adapted_smoothed = MEAN_DIFF_ADAPTED_SMOOTHING_FACTOR * mean_diff_adapted_smoothed + (1 - MEAN_DIFF_ADAPTED_SMOOTHING_FACTOR) * mean_diff_adapted
                    if not vis.win_exists('mean_diff_adapted_smoothed'):
                        vis.line([mean_diff_adapted_smoothed], [batch_num], win='euclid_metric_smoothed', update='append',  opts={**opts, **{'title:': 'euclid_metric_smoothed'}})
                    vis.line([mean_diff_adapted_smoothed], [batch_num], win='euclid_metric_smoothed', update='append')
                    vis.scatter(mu.cpu().detach().numpy(), win='mu_scatter')
                    #vis.scatter(mu.cpu().detach().numpy(), win='mu_scatter_clip')
                    #vis.image(x_recon[0].cpu().detach().numpy(), win='x_recon2')
                    #vis.image(x_recon[int(random.random() * self.batch_size)].cpu().detach().numpy().repeat(4,1).repeat(4,2), win='x_recon',
                    #    opts=dict(title='x_recon'))
                    #if batch_num > 1000:
                    #    vis.line([float(recon_loss)], [batch_num], win='recon_loss', update='append')
                    vis.line([float(recon_loss)], [batch_num], win='recon_loss_all', update='append')
                    vis.line([loss_smoothed], [batch_num], win='recon_loss_smoothed', update='append')
                    vis.line([float(beta_vae_loss)], [batch_num], win='loss', update='append', opts={'title': 'loss'})
                    vis.line([outside_boundary_weight], [batch_num], win='outside_boundary_weight', update='append', opts={'title': 'outside_boundary_weight'})
                    if False:
                        vis.line([float(flattening_loss)], [batch_num], win='flattening_loss', update='append', opts={'title': 'flattening_loss'})
                        FLATTENING_LOSS_SMOOTHED_DISCOUNT = 0.9
                        flattening_loss_smoothed = FLATTENING_LOSS_SMOOTHED_DISCOUNT * flattening_loss_smoothed + (1 - FLATTENING_LOSS_SMOOTHED_DISCOUNT) * flattening_loss
                        vis.line([float(flattening_loss_smoothed)], [batch_num], win='flattening_loss_smoothed', update='append', opts={'title': 'flattening_loss_smoothed'})
                    #x = np.tile(np.arange(1, 101), (100, 1))
                    #y = x.transpose()
                    #X = np.exp((((x - 50) ** 2) + ((y - 50) ** 2)) / -(20.0 ** 2))
                    #vis.surf(X=X, opts=dict(colormap='Hot'), win='surf')
                    if False:
                        euclid_metric = mean_diff_adapted
                        report_dict = {
                                'global_iter': self.global_iter,
                                'recon_loss': float(recon_loss),
                                'total_kld': float(total_kld[0]),
                                'euclid_metric': mean_diff_adapted,
                                }
                        tune.report(report_dict)

                    vis.surf(X=encoding_mus_map, opts=dict(colormap='Hot'), win='surf_mus')
                    if not encoding_mus_map_diff_mean is None:
                        vis.surf(X=encoding_mus_map_diff_mean, opts=dict(colormap='Hot'), win='surf_mus_diff')
                    if not encoding_mus_map_grad is None:
                        #vis.quiver(X=encoding_mus_map_grad, win='encoding_mus_map_grad_quiver')
                        pass

                    if False:
                        X = np.arange(0, .4, .2)
                        Y = np.arange(0, .4, .2)
                        X = np.broadcast_to(np.expand_dims(X, axis=1), (len(X), len(X)))
                        Y = np.broadcast_to(np.expand_dims(Y, axis=0), (len(Y), len(Y)))
                        U = np.multiply(np.cos(X), Y)
                        print("U:", U)
                        V = np.multiply(np.sin(X), Y)
                        print("V:", V)
                        vis.quiver(
                            X=U,
                            Y=V,
                            opts=dict(normalize=0.9),
                            #win='test',
                        )


                    if False:
                        Y = np.linspace(0, 4, 200)
                        win = vis.line(
                            Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2)),
                            X=np.column_stack((Y, Y)),
                            opts=dict(
                                fillarea=True,
                                showlegend=False,
                                width=800,
                                height=800,
                                xlabel='Time',
                                ylabel='Volume',
                                ytype='log',
                                title='Stacked area plot',
                                marginleft=30,
                                marginright=30,
                                marginbottom=80,
                                margintop=30,
                            ),
                            win='test'
                        )



                        Y = np.linspace(0, 4, 200)
                        Y=np.column_stack((np.sqrt(Y), np.sqrt(Y) + 2))
                        X=np.column_stack((Y, Y))
                        #print("X:", X.shape)
                        #vis.line(Y=Y, X=X, win='loss_stacked', update='append', opts={'title': 'flattening_loss'})
                        #vis.line(Y=np.array([[1], [2]]), X=np.array([[batch_num], [batch_num]]), win='loss_stacked', update='append', opts={'title': 'flattening_loss'})





                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and self.global_iter%self.gather_step == 0:
                    # print('=================================================')
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item()))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B':
                        pbar.write('C:{:.3f}'.format(C.data[0]))

                    if self.viz_on:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=F.sigmoid(x_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.gather.flush()

                    if self.viz_on or self.save_output:
                        self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    # self.save_args('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))
                    # self.save_args(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

#    def create_grid_pattern_of_points(self):
#        '''Creates a grid pattern of coordinates that are
#        all stored in a single list.''
#        The grid pattern is created using two for loops.
#        '''                            
#        '''
#        The first for loop creates the x coordinates.
#        The second for loop creates the y coordinates.
#        The two for loops are nested.
#        '''
#        x_coords = []
#        y_coords = []
#        for x in range(self.grid_size):
#            for y in range(self.grid_size):
#                x_coords.append(x)
#        y_coords = []
#        for y in range(self.grid_size):
#            for x in range(self.grid_size):
#                y_co:ords.append(y)
#        #print("x_coords:", x_coords)
#        #print("y_coords:", y_coords)
#        #print("len(x_coords):", len(x_coords))
#        #print("

    def viz_reconstruction_2(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        #images = x_recon
        #self.viz.images(images, env=self.viz_name+'_reconstruction',
        #                opts=dict(title=str(self.global_iter)), nrow=10)
        vis.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        vis.images(images,
                win='reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))


        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


class VAEWrapped():
    def __init__(self):
        net = BetaVAE_H
        self.z_dim = 2
        self.nc = 1
        self.use_cuda = True and torch.cuda.is_available()
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda) 
        self.load_weights()
        self.img_size = 84

    def load_weights(self):
        checkpoint = torch.load('vae2/checkpoints/main/last')
        self.net.load_state_dict(checkpoint['model_states']['net'])

    def encode(self, x):
        x = x.unsqueeze(0)
        latent_dim_vals = self.net._encode(x)/100
        return latent_dim_vals[0][:self.z_dim], latent_dim_vals[0][self.z_dim:]

    def decode(self, z):
        return self.net._decode(z)

    def reparameterize(self, mu, logvar):
        return reparametrize(mu, logvar)

    def forward(self, x):
        return self.net.forward(x)

    def format(self, rgb_array):
        data = torch.from_numpy(rgb_array).float().to(device='cuda')
        #data = data[torch.newaxis,]
        #data = cv2.resize(data, dsize=(64, 64))
        data = data.unsqueeze(0)
        data = data.permute([0, 3, 1, 2])
        data = data[:, 0, :, :]
        data /= 255
        data[data != 1] = 0
        return data

        data = torch.from_numpy(rgb_array).float().to(device='cuda')
        data /= 255
        data = data.permute([2, 0, 1])
        data = data.reshape([-1, 3, self.img_size, self.img_size])
        return data.reshape(-1, self.img_size * self.img_size * 3)





