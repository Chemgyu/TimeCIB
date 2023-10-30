"""

Script to train TimeCIB.

"""

import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random
import argparse

import wandb

from models import VAE, HI_VAE, GP_VAE, TimeCIB
from models import DiagonalEncoder, JointEncoder, BandedJointEncoder, RNNEncoder
from models import ImagePreprocessor
from models.util_get_scores import get_imputed_inputs, get_discriminative_score, get_predictive_score
from models.utils import moving_avg

device = torch.device("cuda")
torch.set_num_threads(4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pn", "--projname", dest="projname", action="store", help="Name of the project", required=True)
    parser.add_argument("--rn", "--runname", dest="runname", action="store", help="Name of the run name", required=True)
    parser.add_argument("--d", "--dataset", dest="dataset", action="store", help="dataset that will be used.", choices=["hmnist", "rotated", "physionet", "beijing", "weather"], required=True)
    parser.add_argument("--m", "--model", dest="model_type", action="store", choices=["vae", "hivae", "gpvae", "timecib"], default='timecib')
    parser.add_argument("--p", "--prior", dest="prior_type", action="store", choices=["norm", "gp"], required=True)
    parser.add_argument("--e", "--encoder", dest="encoder_type", action="store", choices=["diag", "joint", "band", "rnn"], required=True)
    parser.add_argument("--l", "--lamda", dest="lamda", action="store", default=1.0, type=float)
    parser.add_argument("--b", "--beta", dest="beta", action="store", default=1.0, type=float)
    parser.add_argument("--dim", dest="latent_dim", action="store", default=32, type=int)
    parser.add_argument("--s", "--sim_type", dest="sim_type", action="store", choices=["cauchy", "period", "uniform"], default="uniform")
    parser.add_argument("--dir", dest="dir", action="store", default="", type=str)

    parser.add_argument("--clen", dest="cont_length_scale", action="store", default=2.0, type=float)
    parser.add_argument("--cper", dest="cont_period_scale", action="store", default=24.0, type=float)
    parser.add_argument("--conf", dest="cont_conf", action="store", default="False")
    parser.add_argument("--imputed", dest="imputed", action="store", default="None", choices=["None", "forward", "mean"])
    parser.add_argument("--missingtype", dest="missingtype", action="store", default="mnar", choices=["mnar", "spatial", "random", "temporal_neg", "temporal_pos"])
    parser.add_argument("--missing_ratio", dest="missingratio", action="store", default=None)

    parser.add_argument("--t", "--temperature", dest="temperature", action="store", default=1.0, type=float)
    parser.add_argument("--ep", "--epoch", dest="num_epoch", action="store", default=30, type=int)
    parser.add_argument("--seed", dest="seed", action="store", default=0, type=int)
    parser.add_argument("--test", dest="test", action="store", default=False, type=bool)
    args = parser.parse_args()
    
    ###################################
    #   Define universal parameters   #
    ###################################
    args.weight_decay    = getattr(args, "weight_decay", 1e-5)
    args.gradient_clip   = getattr(args, "gradient_clip", 1e5)
    args.print_interval  = getattr(args, "print_interval", 1000)
    args.return_parts    = getattr(args, "return_parts", True)
    args.device          = getattr(args, "device", device)
    
    args.cnn_kernel_size = getattr(args, "cnn_kernel_size", 3)
    args.cnn_sizes       = getattr(args, "cnn_sizes", [256])
    args.testing         = getattr(args, "testing", False) # Use actual test set or not.

    args.kernel          = getattr(args, "kernel", "cauchy") # "rbf", "diffusion", "matern", "cauchy"
    args.kernel_scales   = getattr(args, "kernel_scales", 1) # Number of different length scales sigma for the GP prior: Ignored if model_type is not gp-vae

    args.normalize       = getattr(args, "normalize", True)  # Normalize vectors before contrasting.
    args.cont_sigma      = getattr(args, "cont_sigma", 1.0)    

    ###################################
    # Define data specific parameters #
    ###################################
    if args.dataset == "hmnist":
        args.batch_size      = getattr(args, "batch_size", 64) #64
        if args.missingratio == None: args.datadir = getattr(args, "datadir", f"../data/hmnist/hmnist_{args.missingtype}.npz")
        else: args.datadir = getattr(args, "datadir", f"../data/hmnist/hmnist_random_{args.missingratio}.npz")
        args.data_dim    = getattr(args, "data_dim", 784)
        args.binary      = getattr(args, "binary", True)
        args.time_length = getattr(args, "time_length", 10)
        args.num_classes = getattr(args, "num_classes", 10)
        args.image_shape = getattr(args, "image_shape", (28, 28, 1))

        args.encoder_sizes = getattr(args, "encoder_sizes", [256, 256])
        args.decoder_sizes = getattr(args, "decoder_sizes", [256, 256, 256])
        args.window_size = getattr(args, "window_size", 3)
        args.sigma       = getattr(args, "sigma", 1.0)

        args.length_scale = getattr(args, "length_scale", 2.0) # Length scale value for the GP prior: Ignored if model_type is not gp-vae
        from models.models import BernoulliDecoder
        args.decoder     = getattr(args, "decoder", BernoulliDecoder)
        args.learning_rate   = getattr(args, "learning_rate", 1e-3)
    
    elif args.dataset == "physionet":
        args.batch_size      = getattr(args, "batch_size", 256) #128
        if args.missingratio == None: args.datadir = getattr(args, "datadir", "../data/physionet/physionet.npz")
        else: args.datadir = getattr(args, "datadir", f"../data/physionet/physionet_random_{args.missingratio}.npz")
        args.data_dim    = getattr(args, "data_dim", 35)
        args.binary      = getattr(args, "binary", False)
        args.time_length = getattr(args, "time_length", 48)
        args.num_classes = getattr(args, "num_classes", 2)
        args.image_shape = getattr(args, "image_shape", None)
        args.encoder_sizes = getattr(args, "encoder_sizes", [128, 128])
        args.decoder_sizes = getattr(args, "decoder_sizes", [256, 256])
        args.window_size = getattr(args, "window_size", 24)
        args.sigma       = getattr(args, "sigma", 1.005)
        args.length_scale = getattr(args, "length_scale", 7.0) # Length scale value for the GP prior: Ignored if model_type is not gp-vae
        from models.models import GaussianDecoder
        args.decoder     = getattr(args, "decoder", GaussianDecoder)
        args.learning_rate   = getattr(args, "learning_rate", 1e-3)

    elif args.dataset == "beijing":
        args.batch_size  = getattr(args, "batch_size", 64)
        args.datadir     = getattr(args, "datadir", "../data/beijing/beijing.npz")
        args.data_dim    = getattr(args, "data_dim", 132)
        args.binary      = getattr(args, "binary", False)
        args.time_length = getattr(args, "time_length", 24)
        args.num_classes = getattr(args, "num_classes", 0)
        args.image_shape = getattr(args, "image_shape", None)
        args.encoder_sizes = getattr(args, "encoder_sizes", [128, 128])
        args.decoder_sizes = getattr(args, "decoder_sizes", [256, 256])
        args.window_size = getattr(args, "window_size", 12)
        args.sigma       = getattr(args, "sigma", 1.005)
        args.length_scale = getattr(args, "length_scale", 2.0) # Length scale value for the GP prior: Ignored if model_type is not gp-vae
        from models.models import GaussianDecoder
        args.decoder     = getattr(args, "decoder", GaussianDecoder)
        args.learning_rate   = getattr(args, "learning_rate", 1e-3)
        
    elif args.dataset == "weather":
        args.batch_size  = getattr(args, "batch_size", 16)
        args.datadir     = getattr(args, "datadir", "../data/weather/weather.npz")
        args.data_dim    = getattr(args, "data_dim", 12)
        args.binary      = getattr(args, "binary", False)
        args.time_length = getattr(args, "time_length", 168)
        args.num_classes = getattr(args, "num_classes", 0)
        args.image_shape = getattr(args, "image_shape", None)
        args.encoder_sizes = getattr(args, "encoder_sizes", [128, 128])
        args.decoder_sizes = getattr(args, "decoder_sizes", [256, 256])
        args.window_size = getattr(args, "window_size", 3)
        args.sigma       = getattr(args, "sigma", 1.000)
        args.length_scale = getattr(args, "length_scale", args.cont_length_scale) # Length scale value for the GP prior: Ignored if model_type is not gp-vae
        from models.models import GaussianDecoder
        args.decoder     = getattr(args, "decoder", GaussianDecoder)
        args.learning_rate   = getattr(args, "learning_rate", 2e-4)
        
    elif args.dataset == "rotated":
        args.batch_size      = getattr(args, "batch_size", 64) #64
        args.datadir     = getattr(args, "datadir", "../data/rotated/rotated.npz")
        args.data_dim    = getattr(args, "data_dim", 784)
        args.binary      = getattr(args, "binary", True)
        args.time_length = getattr(args, "time_length", 10)
        args.num_classes = getattr(args, "num_classes", 10)
        args.image_shape = getattr(args, "image_shape", (28, 28, 1))
        args.encoder_sizes = getattr(args, "encoder_sizes", [256, 256])
        args.decoder_sizes = getattr(args, "decoder_sizes", [256, 256, 256])
        args.window_size = getattr(args, "window_size", 5)
        args.sigma       = getattr(args, "sigma", 1.0)
        args.length_scale = getattr(args, "length_scale", 2.0) # Length scale value for the GP prior: Ignored if model_type is not gp-vae
        from models.models import BernoulliDecoder
        args.decoder     = getattr(args, "decoder", BernoulliDecoder)
        args.learning_rate   = getattr(args, "learning_rate", 1e-3)

    else:
        raise ValueError("Dataset must be one of ['hmnist', 'rotated', 'sprites', 'physionet']")

    #############
    # Fix seeds #
    #############
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)

    #############
    # Load data #
    #############
    from dataset import UnifiedDataset
    train_dataset = UnifiedDataset(train=True, test=False, args=args)
    valid_dataset = UnifiedDataset(train=False, test=False, args=args)
    test_dataset  = UnifiedDataset(train=False, test=True, args=args)

    ################
    # Build models #
    ################
    image_preprocessor = None
    if args.dataset in ["hmnist", "rotated"]: image_preprocessor = ImagePreprocessor(args)
    elif args.dataset in ["physionet", "beijing", "weather"]: pass
    else: raise ValueError("Dataset must be one of ['hmnist', 'rotated', 'physionet', 'beijing', 'weather'")
    args.image_preprocessor = getattr(args, "image_preprocessor", image_preprocessor)

    if args.encoder_type == "diag": args.encoder = getattr(args, "encoder", DiagonalEncoder)
    elif args.encoder_type == "joint": args.encoder = getattr(args, "encoder", JointEncoder)
    elif args.encoder_type == "band": args.encoder = getattr(args, "encoder", BandedJointEncoder)
    elif args.encoder_type == "rnn": args.encoder = getattr(args, "encoder", RNNEncoder)
    else: raise ValueError("Encoder type must be one of ['diag', 'joint', 'band', 'rnn']")

    if args.prior_type == "gp": args.transpose = getattr(args, "transpose", True)
    else: args.transpose = getattr(args, "transpose", False)

    model = None
    if args.model_type == "vae": model = VAE(args)
    elif args.model_type == "hivae": model = HI_VAE(args)
    elif args.model_type == "gpvae": model = GP_VAE(args)
    elif args.model_type == "timecib": model = TimeCIB(args)
    else: raise ValueError("Model type must be one of ['vae', 'hivae', 'gpvae', 'timecib']")
    # Retreive the pretrained model, if exists.
    if args.dir != "": model.load_state_dict(torch.load(args.dir+"/files/model_best.pth"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    wandb.init(project=args.projname, name=args.runname, config=args)

    ##################
    # Start training #
    ##################
    best_valid_mse = 1e+3
    best_valid_renew = False
    for epoch in range(args.num_epoch):
        if not args.test:
            print("--------------------")
            print(f"Current Epoch: {epoch+1} / {args.num_epoch}")

            if epoch==0:
                idx, batch = next(enumerate(test_dataset.loader))
                x_full, x_miss, m_miss, m_artificial, y, t = batch
                np.save(os.path.join(wandb.run.dir, f"x_full"), x_full)
                np.save(os.path.join(wandb.run.dir, f"x_miss"), x_miss)
                np.save(os.path.join(wandb.run.dir, f"m_miss"), m_miss)
                x_miss, m_miss, t = x_miss.to(device), m_miss.to(device), t.to(device)
                (x_z, x_z_nott) = model.sample(x_miss, t, m_miss)
                x_z, x_z_nott = x_z.detach().cpu().numpy(), x_z_nott.detach().cpu().numpy()
                np.save(os.path.join(wandb.run.dir, f"x_0"), x_z)
                np.save(os.path.join(wandb.run.dir, f"x_0_nott"), x_z_nott)

            ### Train
            model.train()
            train_loss, train_nll, train_kl, train_smi, train_mae, train_mse, train_rmse, train_mre, train_nll, train_mnll, train_auroc, num_samples, num_missing = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for idx, batch in enumerate(tqdm(train_dataset.loader)):
                x_full, x_miss, m_miss, m_artificial, y, t= batch

                batch_size = len(x_full)
                num_samples += batch_size
                curr_missing = torch.sum(m_artificial).item()
                num_missing += curr_missing

                x_miss, x_full, m_miss, m_artificial, t = x_miss.to(device), x_full.to(device), m_miss.to(device), m_artificial.to(device), t.to(device)
                loss, nll, kl, smi, mae, mse, rmse, mre, mnll = model(x_miss, x_full, m_miss, m_artificial, t)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()

                train_loss, train_nll, train_kl, train_smi, train_mae, train_mse, train_rmse, train_mre, train_mnll = moving_avg(loss, nll, kl, smi, mae, mse, rmse, mre, mnll, train_loss, train_nll, train_kl, train_smi, train_mae, train_mse, train_rmse, train_mre, train_mnll, num_samples, batch_size, num_missing, curr_missing)

                if idx % args.print_interval == 0 and idx > 0:
                    if args.num_classes > 0: print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")
                    else: print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")

            print(f"Train step {idx} | Loss: {train_loss:.3f} | NLL:{train_nll:.3f} | KL:{train_kl:.3f} | SMI:{train_smi:.1f} | MAE:{train_mae:.4f} | MSE:{train_mse:.4f} | RMSE:{train_rmse:.4f} | MRE:{train_mre:.4f} | MNLL:{train_mnll:.4f}")
            wandb.log({"train_loss":train_loss, "train_nll":train_nll, "train_kl":train_kl, "train_smi": train_smi, "train_mae":train_mae, "train_mse":train_mse, "train_rmse":train_rmse, "train_mre":train_mre, "train_mnll": train_mnll, "train_auroc":train_auroc}, step=epoch)

            ### Valid
            model.eval()
            valid_loss, valid_nll, valid_kl, valid_smi, valid_mae, valid_mse, valid_rmse, valid_mre, valid_nll, valid_mnll, valid_auroc, num_samples, num_missing = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for idx, batch in enumerate(tqdm(valid_dataset.loader)):
                x_full, x_miss, m_miss, m_artificial, y, t= batch

                batch_size = len(x_full)
                num_samples += batch_size
                curr_missing = torch.sum(m_artificial).item()
                num_missing += curr_missing

                x_miss, x_full, m_miss, m_artificial, t = x_miss.to(device), x_full.to(device), m_miss.to(device), m_artificial.to(device), t.to(device)
                loss, nll, kl, smi, mae, mse, rmse, mre, mnll = model(x_miss, x_full, m_miss, m_artificial, t, test=True)
                valid_loss, valid_nll, valid_kl, valid_smi, valid_mae, valid_mse, valid_rmse, valid_mre, valid_mnll = moving_avg(loss, nll, kl, smi, mae, mse, rmse, mre, mnll, valid_loss, valid_nll, valid_kl, valid_smi, valid_mae, valid_mse, valid_rmse, valid_mre, valid_mnll, num_samples, batch_size, num_missing, curr_missing)
            if best_valid_mse > valid_mse: best_valid_renew = True
            print(f"Valid step {idx} | Loss: {valid_loss:.3f} | NLL:{valid_nll:.3f} | KL:{valid_kl:.3f} | SMI:{valid_smi:.1f} | MAE:{valid_mae:.4f} | MSE:{valid_mse:.4f} | RMSE:{valid_rmse:.4f} | MRE:{valid_mre:.4f} | MNLL:{valid_mnll:.4f}")
            wandb.log({"valid_loss":valid_loss, "valid_nll":valid_nll, "valid_kl":valid_kl, "valid_smi": valid_smi, "valid_mae":valid_mae, "valid_mse":valid_mse, "valid_rmse":valid_rmse, "valid_mre":valid_mre, "valid_mnll": valid_mnll, "valid_auroc":valid_auroc}, step=epoch)        

            # Save the current results.
            idx, batch = next(enumerate(test_dataset.loader))
            x_full, x_miss, m_miss, m_artificial, y, t = batch
            x_miss, m_miss, t = x_miss.to(device),  m_miss.to(device), t.to(device)
            model.sample_size = args.batch_size
            (x_z, x_z_nott) = model.sample(x_miss, t,  m_miss)
            x_z, x_z_nott = x_z.detach().cpu().numpy(), x_z_nott.detach().cpu().numpy()
            np.save(os.path.join(wandb.run.dir, f"x_{epoch+1}"), x_z)
            np.save(os.path.join(wandb.run.dir, f"x_{epoch+1}_nott"), x_z_nott)
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pth"))

        
        if best_valid_renew or args.test:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model_best.pth"))
            # Get the reconstruction score (Reconstruct each timestep given all timesteps.)
            # print("Getting reconstruction score...")
            test_loss, test_nll, test_kl, test_smi, test_mae, test_mse, test_rmse, test_mre, test_nll, test_mnll, test_auroc, num_samples, num_missing = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            for idx, batch in enumerate(test_dataset.loader):
                x_full, x_miss, m_miss, m_artificial, y, t= batch

                batch_size = len(x_full)
                num_samples += batch_size
                curr_missing = torch.sum(m_artificial).item()
                num_missing += curr_missing

                x_miss, x_full, m_miss, m_artificial, t = x_miss.to(device), x_full.to(device), m_miss.to(device), m_artificial.to(device), t.to(device)
                loss, nll, kl, smi, mae, mse, rmse, mre, mnll = model(x_miss, x_full, m_miss.bool(), m_artificial.bool(), t, test=True) 
                if args.test:
                    mae, mse, rmse, mre = model.get_metrics(x_miss, x_full, m_artificial.bool())
                
                test_loss, test_nll, test_kl, test_smi, test_mae, test_mse, test_rmse, test_mre, test_mnll = moving_avg(loss, nll, kl, smi, mae, mse, rmse, mre, mnll, test_loss, test_nll, test_kl, test_smi, test_mae, test_mse, test_rmse, test_mre, test_mnll, num_samples, batch_size, num_missing, curr_missing)
            print(f"Reconstruction scores: mse:{test_mse:.6f} | nll:{test_mnll:.6f}")
            wandb.log({"reconstruction_mse":test_mse, "reconstruction_nll": test_mnll}, step=epoch)

            x_full, x_imputed, y = get_imputed_inputs(model, test_dataset.loader, device, args.num_classes > 0, args.test)
            # Get the predictive score (Reconstruct the next timestep given previous timesteps.)            
            if args.test or (epoch+1) % 20 == 0: 
                print("Getting predictive score...")
                mae = get_predictive_score(x_full, x_imputed, device, args)
                wandb.log({"predictive_score": mae}, step=epoch)
                print(f"Predictive Score: {mae:.5f}")

            # Get the discriminative score (Classify timestep(hmnist) or timeseries(physionet).)
            if args.num_classes > 0:
                # print("Getting discriminative score...")
                best_valid_renew = False
                auroc, auprc = get_discriminative_score(x_imputed, y, args)
                # if args.dataset == "hmnist": wandb.log({"test_auroc_round":auroc_round, "test_auprc_round":auprc_round}, step=epoch)
                wandb.log({"test_auroc":auroc, "test_auprc":auprc}, step=epoch)
                print(f"Discriminative Score: {auroc:.4f}")
                
        if args.test: break
            
if __name__ == "__main__":
    main()