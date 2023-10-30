import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from models.utils import logmeanexp, gen_mask_t
from models.util_nns import build_nn, build_1d_cnn, build_2d_cnn
from models.util_contrast import cauchy_contrast, sine_contrast, uniform_contrast
from models.util_gp_kernels import rbf_kernel, diffusion_kernel, matern_kernel, cauchy_kernel

from sklearn.metrics import roc_auc_score

######################################################
## Image Preprocessor
######################################################

class ImagePreprocessor(nn.Module):
    def __init__(self, args):
        """
            Preprocessing 2D CNN layer.
            One independent representation per each time step.
            : param image_shape     : input image size. (H, W, C)
            : param cnn_sizes       : hidden channel dimensions of the CNN. eg. [256]
            : param kernel_size     : kernel/filter width and height. eg. 3
        """
        super(ImagePreprocessor, self).__init__()
        self.image_shape = args.image_shape
        self.time_length = args.time_length
        self.net = build_2d_cnn(args.cnn_sizes, self.image_shape[-1], args.cnn_kernel_size)

        for m in self.net.children():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x):
        if self.time_length > 0:
            x = x.reshape(-1, self.image_shape[2], self.image_shape[0], self.image_shape[1])
        x = self.net(x)
        if self.time_length > 0:
            x = x.reshape(-1, self.time_length, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        return x

######################################################
## Encoders
######################################################

class DiagonalEncoder(nn.Module):
    def __init__(self, args):
        """
            Encoder with factorized Normal posterior over temporal dimension
            Used by disjoint VAE and HI-VAE with Standard Normal prior.
            : param data_dim       : input dimension.
            : param output_dim      : output latent dimension.
            : param encoder_sizes     : hidden layer dimensions.
        """
        super(DiagonalEncoder, self).__init__()
        self.input_dim = args.data_dim
        self.output_dim = args.latent_dim
        self.transpose = args.transpose
        # assert self.transpose == False
        
        self.net = build_nn(self.input_dim, args.encoder_sizes, self.output_dim*2)
        self.softplus = nn.Softplus()

        self.prior_type = args.prior_type
        
        for m in self.net.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
        
    def forward(self, x, mask=None):
        mapped = self.net(x) # [B, T, output_dim]
        if self.transpose:
            mapped = mapped.transpose(-1, -2)
            return MultivariateNormal(loc=mapped[:, :self.output_dim, :],
                                      covariance_matrix=torch.diag_embed(self.softplus(mapped[:, self.output_dim:, :])))
        if self.prior_type == "vmf":
            return VonMisesFisher(loc=mapped[:, :, :self.output_dim], scale=self.softplus(mapped[:, :, self.output_dim].unsqueeze(-1)))
        
        return MultivariateNormal(loc=mapped[:, :, :self.output_dim],
                                  covariance_matrix=torch.diag_embed(self.softplus(mapped[:, :, self.output_dim:])))

class JointEncoder(nn.Module):
    def __init__(self, args):
        """
            Encoder with factorized Normal posteior. (Reflecting temporal dependency via 1D CNN)
            Used by joint-VAE and HI-VAE with Standard Normal prior or GP-VAE with factorized Normal posterior
            : param input_dim       : input latent dimension.
            : param output_dim      : output latent dimension.
            : param hidden_dims     : hidden layer dimensions.
            : param kernel_size     : kernel size for Conv1D layer.
            : param transpose       : True for GP prior | False for Standard Normal prior
        """
        super(JointEncoder, self).__init__()
        self.input_dim = args.data_dim
        self.output_dim = args.latent_dim
        self.time_length = args.time_length
        self.transpose = args.transpose
        self.dataset = args.dataset

        self.net_cnn = build_1d_cnn(self.input_dim, args.encoder_sizes[0], args.window_size)
        self.net_nn  = build_nn(args.encoder_sizes[0], args.encoder_sizes, self.output_dim*2)
        self.softplus = nn.Softplus()

        self.prior_type = args.prior_type

        for m in self.net_cnn.children():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
        for m in self.net_nn.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x, mask=None):
        x = torch.transpose(x, -1, -2) 
        x = self.net_cnn(x)  # input: [B, T, D], to use nn.Conv1d
        x = torch.transpose(x, -1, -2)
        mapped = self.net_nn(x) # output: [B, T, 2*H]

        if self.transpose:
            mapped = torch.transpose(mapped, -1, -2)
            if self.dataset in ["weather", "ecl", "etth1", "etth2", "ettm1", "ettm2"]:
                return MultivariateNormal(loc=mapped[:, :self.output_dim, :],
                                      covariance_matrix=torch.diag_embed(self.softplus(mapped[:, self.output_dim:, :]) + torch.ones_like(self.softplus(mapped[:, self.output_dim:, :])) * 1e-7))    
            return MultivariateNormal(loc=mapped[:, :self.output_dim, :],
                                      covariance_matrix=torch.diag_embed(self.softplus(mapped[:, self.output_dim:, :])))
        
        eps = torch.ones_like(self.softplus(mapped[:, :, self.output_dim:])) * 1e-9 # For the numerical stability.
        loc = mapped[:, :, :self.output_dim]
        scale = self.softplus(mapped[:, :, self.output_dim:]) + eps

        return MultivariateNormal(loc=loc, covariance_matrix=torch.diag_embed(scale))

class BandedJointEncoder(nn.Module):
    def __init__(self, args):
        """
            Encoder with multivariate Normal posterior.
            Used by GP-VAE with band covariance matrix.
            : param input_dim       : input latent dimension.
            : param hidden_dims     : hidden layer dimensions.
            : param kernel_size     : kernel size for Conv1D layer.
        """
        super(BandedJointEncoder, self).__init__()
        self.input_dim = args.data_dim
        self.output_dim = args.latent_dim
        self.time_length = args.time_length

        # assert args.transpose==True # Use BandedJointEncoder only for the GP-prior, which requires transpose.
        
        self.task = args.dataset
        
        self.net_cnn = build_1d_cnn(self.input_dim, args.encoder_sizes[0], args.window_size)
        self.net_nn  = build_nn(args.encoder_sizes[0], args.encoder_sizes, self.output_dim*3)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        
        for m in self.net_cnn.children():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
        for m in self.net_nn.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

    def forward(self, x, mask=None):
        x = torch.transpose(x, -1, -2) 
        x = self.net_cnn(x)  # input: [B, T, D], to use nn.Conv1d
        x = torch.transpose(x, -1, -2)
        mapped = self.net_nn(x) # output: [B, T, 3H]

        mapped_mean = mapped[:, :, :self.output_dim]
        mapped_prec = mapped[:, :, self.output_dim:]

        if self.task == "physionet": mapped_prec = self.sigmoid(mapped_prec)
        else: mapped_prec = self.softplus(mapped_prec)

        mean = torch.transpose(mapped_mean, -1, -2)

        mapped_prec_reshaped = mapped_prec.reshape(-1, self.output_dim, 2*self.time_length) # [B, T, 2H] > [B, H, 2T]
        prec_matrix_diag = torch.diag_embed(mapped_prec_reshaped[:, :, :self.time_length])
        prec_matrix_band = torch.diag_embed(mapped_prec_reshaped[:, :, self.time_length:-1], offset=1)
        eye = torch.diag_embed(torch.ones_like(mapped_prec_reshaped[:, :, :self.time_length]))
        prec_matrix = prec_matrix_diag + prec_matrix_band + eye
        scale = torch.linalg.solve_triangular(prec_matrix, eye, upper=False)
        return MultivariateNormal(loc=mean, scale_tril=scale)

class RNNEncoder(nn.Module):
    def __init__(self, args):
        """
            Encoder with RNN encoder.
        """
        super(RNNEncoder, self).__init__()
        self.input_dim = args.data_dim
        self.output_dim = args.latent_dim
        self.time_length = args.time_length
        self.task = args.dataset
        self.transpose = args.transpose
        self.softplus = nn.Softplus()
        self.prior_type = args.prior_type

        self.net_rnn = nn.LSTM(input_size=self.input_dim*2, hidden_size=self.output_dim//2, num_layers=len(args.encoder_sizes),
                               batch_first=True, dropout=0.1, bidirectional=True)
        self.net_nn = build_nn(self.output_dim, [self.output_dim], self.output_dim*2)
        self.position_enc = PositionalEncoding(self.input_dim, self.time_length)

    def forward(self, x, mask=None):
        x = self.position_enc(x)
        x = torch.concatenate((x, mask.float()), dim=-1)
        x, _ = self.net_rnn(x)
        mapped = self.net_nn(x)

        if self.transpose:
            mapped = torch.transpose(mapped, -1, -2)
            eps = torch.ones_like(self.softplus(mapped[:, self.output_dim:, :])) * 1e-7
            loc = mapped[:, :self.output_dim, :]
            cov = torch.diag_embed(self.softplus(mapped[:, self.output_dim:, :])+eps)
            return MultivariateNormal(loc=mapped[:, :self.output_dim, :],
                                    covariance_matrix = cov)
        
        eps = torch.ones_like(self.softplus(mapped[:, :, self.output_dim:])) * 1e-9 # For the numerical stability.
        loc = mapped[:, :, :self.output_dim]
        scale = self.softplus(mapped[:, :, self.output_dim:]) + eps

        return MultivariateNormal(loc=loc, covariance_matrix=torch.diag_embed(scale))

######################################################
## Decoders
######################################################

class Decoder(nn.Module):
    def __init__(self, args):
        """
            Decoder parent class with no specific output distribution.
            :param output_dim   : output dimension.
            :param hidden_dims  : tuple of hidden dimensions.
        """
        super(Decoder, self).__init__()
        self.input_dim = args.latent_dim
        self.hidden_dims = args.decoder_sizes
        self.output_dim = args.data_dim
        self.net = build_nn(self.input_dim, args.decoder_sizes, self.output_dim)
        self.dataset = args.dataset
        for m in self.net.children():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)
    
    def forward(self, x):
        pass

class BernoulliDecoder(Decoder):
    """
        Decoder child class with Bernoulli output distribution (used for image datasets).
    """
    def forward(self, x):
        mapped = self.net(x)
        mapped = torch.nan_to_num(mapped)
        return torch.distributions.bernoulli.Bernoulli(logits=mapped)

class GaussianDecoder(Decoder):
    """
        Decoder child class with Gaussian output distribution.
    """
    def forward(self, x):
        mean = self.net(x)
        var = torch.ones_like(mean)
        return torch.distributions.normal.Normal(loc=mean, scale=var)

######################################################
## VAE Models
######################################################

class VAE(nn.Module):
    def __init__(self, args):
        """
            Basic Variational AutoEncoder with Standard Normal prior.
            : param hidden_dim          : hidden dimension.
            : param input_dim           : input dimension.
            : param time_length         : time series duration.

            : param encoder             : encoder model class {Diagonal, Joint, BandedJoint}
            : param encoder_sizes       : layer sizes for the encoder.
            : param decoder             : decoder model class {Bernoulli, Gaussian}.
            : param decoder_sizes       : layer sizes for the decoder.
            
            : param image_preprocessor  : 2D-CNN for image preprocessing.
            : param beta                : tradeoff coefficient between reconstruction and KL-terms in ELBO.
            : param M                   : number of Monte Carlo samples for ELBO estimation.
            : param K                   : number of importance weights for IWAE model (https://arxiv.org/abs/1509.00519).
        """
        super(VAE, self).__init__()
        self.hidden_dim = args.latent_dim
        self.input_dim = args.data_dim
        self.time_length = args.time_length
        self.transpose = args.transpose
        self.image_shape = args.image_shape
        self.device = args.device
    
        self.model_type = args.model_type
        self.dataset = args.dataset
        self.encoder = args.encoder(args)
        self.decoder = args.decoder(args)
        self.preprocessor = args.image_preprocessor
        
        self.batch_size = args.batch_size
        self.sample_size = self.batch_size # only used for the first sample call.
        self.old_sample_size = 0

        self.prior = None
        self.prior_type = args.prior_type
        self.beta = args.beta
        self.lamda = args.lamda
        self.binary = args.binary
        self.num_classes = args.num_classes
        self.t = self.time_length // 2
        self.test = args.test

        self.epsilon = 1e-7
        self.temperature = args.temperature
        self.normalize = args.normalize
        self.cont_sigma = args.cont_sigma
        self.cont_length_scale = args.cont_length_scale
        self.cont_period_scale = args.cont_period_scale
        self.cont_conf = args.cont_conf
        self.sim_type = args.sim_type

        self.return_parts = args.return_parts
    
    def encode(self, x, m_mask, m_exist):
        ## get q(Z_t | X^o_{1:T}) and q(Z_t | X^o_\t).
        if self.model_type == 'timecib':
            x_ = x.reshape(self.batch_size, self.time_length, self.input_dim) # [B, T, D]
            # Choose a random timepoint that will be masked out.
            x_nott = x.clone()
            self.t = np.random.randint(0, self.time_length)                 
            m_nott = torch.tile(gen_mask_t(self.t, self.time_length), (self.batch_size, 1, self.input_dim)).to(self.device)
            m_exist_nott = (m_exist.float() - m_nott[:,:,0].float()).bool()
            m_mask_nott = (m_mask.float() + m_nott.float()).bool()
            x_replace = torch.zeros_like(x_nott)
            x_nott = torch.where(m_nott, x_replace, x_nott) # [B, T, D], the same shape of x.
        else:
            # For other VAE models.
            x_nott, m_exist_nott = x, m_exist

        if self.preprocessor is not None:
            # For the image datasets - HMNIST and SPRITES.
            if self.stochastic: x_nott = x_nott.reshape(self.batch_size, self.time_length, self.image_shape[0], self.image_shape[1], self.image_shape[2])
            else: x_nott = x_nott.reshape(self.batch_size * self.time_length, self.time_length, self.image_shape[0], self.image_shape[1], self.image_shape[2])
            x, x_nott = self.preprocessor(x), self.preprocessor(x_nott)
            x, x_nott = x.reshape(-1, self.time_length, self.input_dim), x_nott.reshape(-1, self.time_length, self.input_dim)
        
        return self.encoder(x, m_mask), self.encoder(x_nott, m_mask_nott)

    def decode(self, z):
        if self.transpose: z = torch.transpose(z, -1, -2)
        return self.decoder(z)

    def get_prior(self):
        # Gaussain uniform prior: N(0, I).
        assert self.prior_type == "norm"
        assert self.transpose == False
        if not self.sample_size == self.old_sample_size:
            # For the corner case of last batch that changes the batch size.
            prior_loc = torch.zeros(self.hidden_dim).to(self.device)
            prior_cov = torch.eye(self.hidden_dim).to(self.device)
            self.prior = MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
            self.old_sample_size = self.sample_size
        return self.prior
    
    def get_prior_full(self):
        prior_loc = torch.zeros(self.input_dim).to(self.device)
        prior_cov = torch.ones_like(prior_loc).to(self.device)
        self.prior_full = torch.distributions.normal.Normal(loc=prior_loc, scale=prior_cov)
        return self.prior_full
    
    def get_prior_data(self):
        # Gaussain uniform prior: N(0, I).
        assert self.prior_type == "norm"
        assert self.transpose == False
            # For the corner case of last batch that changes the batch size.
        prior_loc = torch.zeros(self.input_dim).to(self.device)
        prior_cov = torch.eye(self.input_dim).to(self.device)
        self.prior = MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
        self.old_sample_size = self.sample_size
        return self.prior

    def get_nll(self, x, px_z, m_mask, m_exist):
        ## Calculate the nll for observed features (not m_mask, m_exist)
        nll = -px_z.log_prob(x.reshape(self.sample_size, self.time_length, self.input_dim)) # shape = (M*K*BS, TL, D)
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        if self.model_type not in ["vae"]: # m_mask is not None: hivae, gpvae, vmfvae, mis
            nll = torch.where(m_mask, torch.zeros_like(nll), nll) # [M*K*BS, TL, D]
        nll = torch.sum(nll, dim=2) # [M*K*BS, TL]
        nll = torch.where(m_exist, nll, torch.zeros_like(nll))
        nll = torch.sum(nll) / torch.sum(m_exist)
        return nll

    def get_nll_missing(self, x_full, px_z, m_mask):
        ## Calculate the nll only for the artificially masked features.
        nll = -px_z.log_prob(x_full.reshape(self.sample_size, self.time_length, self.input_dim))
        nll = torch.where(torch.isfinite(nll), nll, torch.zeros_like(nll))
        nll = torch.where(m_mask, nll, torch.zeros_like(nll))
        nll = torch.sum(nll) / torch.sum(m_mask) # [M*K*BS]
        return nll
    
    def get_metrics(self, x_z_mean, x_full, m_mask):
        ## Calculate the reconstruction metrics for artificially masked features.
        if self.binary: x_z_mean = torch.round(x_z_mean)
        error = (torch.abs(x_z_mean - x_full).reshape(self.sample_size, self.time_length, self.input_dim))
        error = torch.where(m_mask, error, torch.zeros_like(error))
        # error = torch.where(torch.tile(m_exist.unsqueeze(-1), (1,1,self.input_dim)), error, torch.zeros_like(error))

        mae = torch.sum(error)/ torch.sum(m_mask)
        mse = torch.sum(torch.square(error)) / torch.sum(m_mask)
        rmse = torch.sqrt(mse)
        mre = torch.sum(error) / torch.sum(torch.abs(m_mask * (x_full.reshape(m_mask.shape))))
        
        return mae, mse, rmse, mre

    def sample_classify(self, x, t, mask=None):
        self.batch_size = len(x)
        m_exist = (t >= 0).bool()
        qz_x, _ = self.encode(x, mask, m_exist)
        z = qz_x.mean
        px_z = self.decode(z)
        x_z = px_z.mean
        return x_z

    def sample(self, x, t, mask=None):
        self.batch_size = len(x)
        m_exist = (t >= 0).bool()
        shape = x.shape
        pz = self.get_prior()
        qz_x, qz_x_nott = self.encode(x, mask, m_exist)
        
        z = qz_x.rsample() 
        z_nott = qz_x_nott.rsample()
        
        px_z = self.decode(z)
        px_z_nott = self.decode(z_nott)

        x_z = px_z.mean
        x_z_nott = px_z_nott.mean
        # if self.binary: x_z = torch.round(x_z)
        x_z = x_z.reshape(shape)
        # if self.cont_type in ["sum", "product"]: 
        #     if self.preprocessor is not None:
        #         x_z_nott = x_z_nott.reshape(self.batch_size * self.time_length, self.time_length, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        # else: 
        x_z_nott = x_z_nott.reshape(shape)        
        
        return x_z, x_z_nott

    def get_smi(self, z, z_nott, m_exist, m_mask):
        """ Series Mutual Information """
        # z: [B, T, D] z_nott : [B*T, T, D]
        if self.transpose: z = z.transpose(-1, -2)
        if self.normalize: z, z_nott = F.normalize(z, p=2, dim=-1), F.normalize(z_nott, p=2, dim=-1)

        z_anchor = z[:, self.t, :] #[B, D]
        #alignment
        t_mask = torch.LongTensor([i for i in range(self.t)] + [i+self.t+1 for i in range(self.time_length - self.t - 1)]).to(self.device)
        z_nott = torch.index_select(z_nott,1,t_mask).reshape(-1, self.time_length-1, self.hidden_dim) #[B, T-1, D] 
        sim_align = torch.bmm(z_anchor.unsqueeze(-2), z_nott.transpose(1, 2)).squeeze() #[B, 1, T-1] > [B, T-1]

        # uniformity
        z_unifo = torch.index_select(z, 1, t_mask).reshape(-1, self.hidden_dim) #[B, T-1, D] > [B(T-1), D]
        sim_unifo = torch.matmul(z_anchor, z_unifo.transpose(0,1)) #[B, B(T-1)]
                                                                                        
        # kernel coefficients
        if self.sim_type == "cauchy":  sim_coef = cauchy_contrast(self.time_length, self.cont_sigma, self.cont_length_scale)
        elif self.sim_type == "period":  sim_coef = sine_contrast(self.time_length, self.cont_sigma, self.cont_length_scale, self.cont_period_scale)
        elif self.sim_type == "uniform": sim_coef = uniform_contrast(self.time_length)
        else: sim_coef = uniform_contrast(self.time_length)
        sim_coef = torch.tile(torch.index_select(sim_coef[self.t].to(self.device), 0, t_mask).unsqueeze(0), (self.batch_size, 1)) #[B,T-1]
        sim_coef = sim_coef * (self.time_length-1) / torch.sum(sim_coef, dim=-1, keepdim=True)
        
        sim_max = torch.max(sim_unifo)
        exp_align = torch.exp((sim_align - sim_max) / self.temperature) * sim_coef
        exp_unifo = torch.exp((sim_unifo - sim_max) / self.temperature)
        align = torch.sum(exp_align, dim=-1)
        unifo = torch.sum(exp_unifo, dim=-1)
        logu = torch.log(align / unifo)
        
        return torch.sum(logu)
    
    def forward(self, x, x_full, m_mask, m_artificial=None, t=None, test=False):
        self.batch_size = len(x)
        self.sample_size = self.batch_size
        m_exist = (t >= 0).bool()

        if m_mask is not None: m_mask = m_mask.bool().reshape(self.sample_size, self.time_length, self.input_dim)
        if m_artificial is not None: m_artificial = m_artificial.bool().reshape(self.sample_size, self.time_length, self.input_dim)

        pz = self.get_prior()
        qz_x, qz_nott = self.encode(x, m_mask, m_exist)
        if test: z = qz_x.mean
        else: z = qz_x.rsample()
        px_z = self.decode(z)

        ### Negative Log Likelihood ###
        nll = self.get_nll(x, px_z, m_mask, m_exist)

        ### ELBO with KL###
        kl = torch.distributions.kl.kl_divergence(qz_x, pz) #[M*K*BS, TL or d]
        kl = torch.where(torch.isfinite(kl), kl, torch.zeros_like(kl))
        if not self.transpose: kl = torch.where(m_exist, kl, torch.zeros_like(kl))
        kl = torch.sum(kl) / torch.sum(m_exist)

        ### Contrastive Form ###
        z_mean = qz_x.mean
        z_nott_mean = qz_nott.mean
        if not self.transpose: smi = self.get_smi(z_mean, z_nott_mean, m_exist, m_mask)
        else: smi = torch.Tensor([1.0])

        ### loss
        if self.model_type == "timecib": loss = nll + self.beta * kl - self.lamda * smi
        else: loss = nll + self.beta * kl
        loss = torch.mean(loss)

        ### NLL for the missing ###
        if m_artificial == None: m_nll = self.get_nll_missing(x_full, px_z, m_mask)
        else: m_nll = self.get_nll_missing(x_full, px_z, m_artificial)

        ### Metrics ###
        x_z_mean = self.decode(z_mean).mean
        mae, mse, rmse, mre = self.get_metrics(x_z_mean, x_full, m_artificial)
        
        ### Return ###
        if self.return_parts:
            return loss, nll, kl, -smi, mae, mse, rmse, mre, m_nll
        return loss, mse

class HI_VAE(VAE):
    """ HI-VAE model, where the reconstruction term in ELBO is summed only over observed components """
    def compute_loss(self, x, m_mask=None, return_parts=False):
        return self._compute_loss(x, m_mask=m_mask, return_parts=return_parts)

class GP_VAE(VAE):
    def __init__(self, args):
        """
            GP-VAE with Gaussian Process prior.
            :param kernel: Gaussial Process kernel ["cauchy", "diffusion", "rbf", "matern"]
            :param sigma: scale parameter for a kernel function
            :param length_scale: length scale parameter for a kernel function
            :param kernel_scales: number of different length scales over latent space dimensions
        """
        super(GP_VAE, self).__init__(args)
        self.kernel=args.kernel
        self.sigma=args.sigma
        self.cont_sigma=args.cont_sigma
        self.length_scale = args.length_scale
        self.cont_length_scale = args.cont_length_scale
        self.kernel_scales = args.kernel_scales
    
    def get_prior(self):
        assert self.prior_type == "gp"
        assert self.transpose == True
        if not self.sample_size == self.old_sample_size:
            kernel_matrices = []
            for i in range(self.kernel_scales):
                if self.kernel == "rbf":
                    kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "diffusion":
                    kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "matern":
                    kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                elif self.kernel == "cauchy":
                    kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

            tiled_matrices = []
            total = 0
            for i in range(self.kernel_scales):
                if i == self.kernel_scales-1:
                    multiplier = self.hidden_dim - total
                else:
                    multiplier = int(np.ceil(self.hidden_dim / self.kernel_scales))
                    total += multiplier
                tiled_matrices.append(torch.tile(torch.unsqueeze(kernel_matrices[i], 0), (multiplier, 1, 1)))
            kernel_matrix_tiled = torch.concatenate(tiled_matrices)
            assert len(kernel_matrix_tiled) == self.hidden_dim

            mean = torch.zeros([self.sample_size, self.hidden_dim, self.time_length]) 
            cov  = torch.tile(kernel_matrix_tiled, (self.sample_size, 1, 1, 1))

            self.prior = MultivariateNormal(loc=mean.to(self.device), covariance_matrix=cov.to(self.device))
            self.old_sample_size = self.sample_size
        return self.prior
    
class TimeCIB(VAE):
    def __init__(self, args):
        super(TimeCIB, self).__init__(args)
        self.lamda = args.lamda
        
    def get_prior(self):
        if self.prior_type == "norm":
            if not self.sample_size == self.old_sample_size:
                prior_loc = torch.zeros(self.hidden_dim).to(self.device)
                prior_cov = torch.eye(self.hidden_dim).to(self.device)
                self.prior = MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)
                self.old_sample_size = self.sample_size

        elif self.prior_type == "gp":
            assert self.transpose == True
            if not self.sample_size == self.old_sample_size:
                kernel_matrices = []
                for i in range(self.kernel_scales):
                    if self.kernel == "rbf":
                        kernel_matrices.append(rbf_kernel(self.time_length, self.length_scale / 2**i))
                    elif self.kernel == "diffusion":
                        kernel_matrices.append(diffusion_kernel(self.time_length, self.length_scale / 2**i))
                    elif self.kernel == "matern":
                        kernel_matrices.append(matern_kernel(self.time_length, self.length_scale / 2**i))
                    elif self.kernel == "cauchy":
                        kernel_matrices.append(cauchy_kernel(self.time_length, self.sigma, self.length_scale / 2**i))

                tiled_matrices = []
                total = 0
                for i in range(self.kernel_scales):
                    if i == self.kernel_scales-1:
                        multiplier = self.hidden_dim - total
                    else:
                        multiplier = int(np.ceil(self.hidden_dim / self.kernel_scales))
                        total += multiplier
                    tiled_matrices.append(torch.tile(torch.unsqueeze(kernel_matrices[i], 0), (multiplier, 1, 1)))
                kernel_matrix_tiled = torch.concatenate(tiled_matrices)
                assert len(kernel_matrix_tiled) == self.hidden_dim

                mean = torch.zeros([self.sample_size, self.hidden_dim, self.time_length]) 
                cov  = torch.tile(kernel_matrix_tiled, (self.sample_size, 1, 1, 1))

                self.prior = MultivariateNormal(loc=mean.to(self.device), covariance_matrix=cov.to(self.device))
                self.old_sample_size = self.sample_size
        else: raise ValueError("Prior type must be one of ['norm', 'vMF', 'gp']")
        return self.prior
    
####
# Positional Encoding #
####
class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer(
            "pos_table", self._get_sinusoid_encoding_table(n_position, d_hid)
        )

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""

        def get_position_angle_vec(position):
            return [
                position / np.power(10000, 2 * (hid_j // 2) / d_hid)
                for hid_j in range(d_hid)
            ]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()