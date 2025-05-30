
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import tqdm
import wandb
from info_theory_experiments.models import DecoupledSmileMIEstimator, DownwardSmileMIEstimator, GeneralSmileMIEstimator
from einops import reduce


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_feature_network(
    config,
    trainloader,
    feature_network_training, 
    project_name: str,
    feature_network_A = None,
    model_dir_prefix = None
):
    """
    - Main function for training and evaluating the feature_network.
    - In training mode, the function will train either a model_A or a model_B.
    - In evaluation mode, the feature network will be frozen and adjusted_Psi will be calculated for the network.
    - If in eval_mode, no feature_network_A should be provided because Psi eval_mode is just for making sure
        the feature learned is emergent, and the MI with another feature_network is not relevant to this.

    - train_model_B mode is a subset of train_mode
    - Psi can optionally be replaced with adjusted_Psi in any mode
    """
    
    valid_dataset_types = ['bits', 'ecog', 'resid', 'FMRI', 'meg', 'gol']
    if config["dataset_type"] not in valid_dataset_types:
        raise ValueError(f"dataset_type must be one of {valid_dataset_types}")
    # make sure that if config['train_model_B'] is True, then feature_network_A is not None, and if False, then it is None, raise error otherwise
    if config['train_model_B']:
        print("<<Training model B>>")
        if feature_network_A is None:
            raise ValueError("feature_network_A must be provided if training model B")
    else:
        if feature_network_A is not None:
            raise ValueError("feature_network_A must be None if not training model B")
    if not config['train_mode']: # if eval_mode => then should be no feature_network_A
        if feature_network_A is not None:
            raise ValueError("feature_network_A must be None if not training model B")        
    
    wandb.init(project=project_name, config=config)

    decoupled_MI_estimator = DecoupledSmileMIEstimator(
        feature_size=config['feature_size'],
        critic_output_size=config['decoupled_critic_config']['critic_output_size'],
        hidden_sizes_1=config['decoupled_critic_config']['hidden_sizes_encoder_1'],
        hidden_sizes_2=config['decoupled_critic_config']['hidden_sizes_encoder_2'],
        clip=config['clip'],
        include_bias=config['decoupled_critic_config']['bias']
        ).to(device)
    downward_MI_estimators = [
        DownwardSmileMIEstimator(
            feature_size=config['feature_size'],
            critic_output_size=config['downward_critics_config']['critic_output_size'],
            hidden_sizes_v_critic=config['downward_critics_config']['hidden_sizes_v_critic'],
            hidden_sizes_xi_critic=config['downward_critics_config']['hidden_sizes_xi_critic'],
            clip=config['clip'],
            include_bias=config['downward_critics_config']['bias']
            ).to(device) 
        for _ in range(config['num_atoms'])
    ]

    decoupled_optimizer = torch.optim.Adam(
        decoupled_MI_estimator.parameters(),
        lr=config['decoupled_critic_config']["lr"],
        weight_decay=config['decoupled_critic_config']["weight_decay"]
    )
    downward_optims = [
        torch.optim.Adam(
            dc.parameters(),
            lr=config['downward_critics_config']["lr"],
            weight_decay=config['downward_critics_config']["weight_decay"]
        ) 
        for dc in downward_MI_estimators
    ]


    # init bit specific estimators
    if config["dataset_type"] == 'bits':
        xor_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=32,
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=True
        ).to(device)
        extra_bit_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=32,
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=True
        ).to(device)
        bonus_bit_estimator = GeneralSmileMIEstimator(
            x_dim=config['feature_size'],
            y_dim=1,
            critic_output_size=32,
            x_critics_hidden_sizes=[512, 512, 128],
            y_critics_hidden_sizes=[512, 512, 128],
            clip=config['clip'],
            include_bias=True
        ).to(device)
        extra_bit_optimizer = torch.optim.Adam(
            extra_bit_estimator.parameters(),
            lr=1e-4,
            weight_decay=0
        )

        bonus_bit_optimizer = torch.optim.Adam(
            bonus_bit_estimator.parameters(),
            lr=1e-4,
            weight_decay=0
        )
        xor_optimizer = torch.optim.Adam(
            xor_estimator.parameters(),
            lr=1e-4,
            weight_decay=0
        )


    # init feature network optimizer
    if config['train_mode']:
        feature_optimizer = torch.optim.Adam(
            feature_network_training.parameters(),
            lr=config['feature_network_config']["lr"],
            weight_decay=config['feature_network_config']["weight_decay"]
        )
        # init train_model_B specific estimators
        if config['train_model_B']:
            MI_AB_estimator = GeneralSmileMIEstimator(
                x_dim=config['feature_size'],
                y_dim=config['feature_size'],
                critic_output_size=config['downward_critics_config']['critic_output_size'],
                x_critics_hidden_sizes=[512, 512, 128],
                y_critics_hidden_sizes=[512, 512, 128],
                clip=config['clip'],
                include_bias=True
            ).to(device)
            MI_AB_optimizer = torch.optim.Adam(
                MI_AB_estimator.parameters(),
                lr=1e-3,
                weight_decay=0
            )

    wandb.watch(feature_network_training, log='all')

    ##
    ## TRAINING LOOP
    ##

    epochs = config['epochs']

    step = 0
    for _ in tqdm.tqdm(range(epochs), desc='Training'):
        for batch_num, batch in enumerate(trainloader):
            x0 = batch[:, 0].to(device).float().unsqueeze(1)
            x1 = batch[:, 1].to(device).float().unsqueeze(1)
            v0_B = feature_network_training(x0).detach()
            v1_B = feature_network_training(x1).detach()
            if config['train_model_B']:
                v0_A = feature_network_A(x0).detach()


            # update decoupled critic
            decoupled_optimizer.zero_grad()
            decoupled_MI = decoupled_MI_estimator(v0_B, v1_B)
            decoupled_loss = -decoupled_MI
            decoupled_loss.backward(retain_graph=True)
            decoupled_optimizer.step()


            # update each downward critic 
            x0_flat = x0.view(x0.size(0), -1)  # Flatten x0 to (batch_size, 10*10)
            for i in range(config['num_atoms']):
                downward_optims[i].zero_grad()
                channel_i = x0_flat[:, i].unsqueeze(1).detach()
                downward_MI_i = downward_MI_estimators[i](v1_B, channel_i)
                downward_loss = - downward_MI_i
                downward_loss.backward(retain_graph=True)
                downward_optims[i].step()
                wandb.log({
                    f"downward_MI_{i}": downward_MI_i   
                }, step=step)

            # update MI_AB_estimator
            if config['train_model_B']:
                MI_AB_optimizer.zero_grad()
                MI_AB = MI_AB_estimator(v0_B, v0_A)
                MI_AB_loss = -MI_AB
                MI_AB_loss.backward(retain_graph=True)
                MI_AB_optimizer.step()
                wandb.log({"MI_AB": MI_AB},step=step)


            # Find Psi (and maybe adjust it)
            if config["train_mode"]:
                feature_optimizer.zero_grad()

            downward_MIs_post_update = [] # used for calculating the adjusted Psi and Psi adjustment
            v0_B = feature_network_training(x0)
            v1_B = feature_network_training(x1)

            for i in range(config['num_atoms']): # finds sum of downward terms
                channel_i = x0_flat[:, i].unsqueeze(1)
                channel_i_MI = downward_MI_estimators[i](v1_B, channel_i)
                downward_MIs_post_update.append(channel_i_MI)

            sum_downward_MI_post_update = sum(downward_MIs_post_update)
            decoupled_MI_post_update = decoupled_MI_estimator(v0_B, v1_B)

            if config["adjust_Psi"]:
                clipped_min_MIs = max(0, min(downward_MIs_post_update))
                Psi_adjustment = (config['num_atoms'] - 1) * clipped_min_MIs
                Psi = decoupled_MI_post_update - sum_downward_MI_post_update + Psi_adjustment
            else:
                Psi = decoupled_MI_post_update - sum_downward_MI_post_update

            if config['train_mode']: # if in training mode, update the network
                if config['train_model_B']:
                    MI_AB_post_update = MI_AB_estimator(v0_B, v0_A)
                    if config['minimize_neg_terms_until'] < step:

                        feature_loss = - Psi + MI_AB_post_update
                    else:
                        feature_loss = - (-sum_downward_MI_post_update - MI_AB_post_update)
                else:
                    if config['minimize_neg_terms_until'] < step:
                        feature_loss = - Psi
                    else:
                        feature_loss = - (-sum_downward_MI_post_update)

                if config['start_updating_f_after'] < step:
                    if batch_num % config['update_f_every_N_steps'] == 0:
                        feature_loss.backward(retain_graph=True)
                        feature_optimizer.step()

            wandb.log({
                "decoupled_MI": decoupled_MI_post_update,
                "sum_downward_MI": sum_downward_MI_post_update,
                "Psi": Psi,
                # "feature_loss": feature_loss,
            }, step=step)


            if config["dataset_type"] == 'bits':
                v0_B = feature_network_training(x0).detach()
                v1_B = feature_network_training(x1).detach()
                xor_bits = (reduce(x0[: , :5], 'b n -> b', 'sum') % 2).unsqueeze(1)
                extra_bit = x0[:, -1].unsqueeze(1)
                bonus_bit = ( xor_bits + extra_bit ) % 2


                xor_optimizer.zero_grad()
                xor_MI = xor_estimator(v0_B, xor_bits)
                xor_loss = -xor_MI
                xor_loss.backward(retain_graph=True)
                xor_optimizer.step()

                extra_bit_optimizer.zero_grad()
                extra_bit_MI = extra_bit_estimator(v0_B, extra_bit)
                extra_bit_loss = -extra_bit_MI
                extra_bit_loss.backward(retain_graph=True)
                extra_bit_optimizer.step()

                bonus_bit_optimizer.zero_grad()
                bonus_bit_MI = bonus_bit_estimator(v0_B, bonus_bit)
                bonus_bit_loss = -bonus_bit_MI
                bonus_bit_loss.backward(retain_graph=True)
                bonus_bit_optimizer.step()

                wandb.log({
                    "xor_MI": xor_MI,
                    "extra_bit_MI": extra_bit_MI,
                    "bonus_bit_MI": bonus_bit_MI
                }, step=step)

            step += 1

    if model_dir_prefix is not None:
        torch.save(feature_network_training.state_dict(), f"models/{model_dir_prefix}-{wandb.run.name}.pth")

    name = wandb.run.name

    wandb.finish()
    
    return feature_network_training, name

