from info_theory_experiments.custom_datasets import FMRIDatasetConcat
import torch
from info_theory_experiments.trainers import train_feature_network
from info_theory_experiments.models import SkipConnectionSupervenientFeatureNetwork

dataset = FMRIDatasetConcat()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# in this experiment we learn emergent features on the FMRI dataset

for seed in range(10):

    torch.manual_seed(seed)

    config = {
        "torch_seed": seed,
        "dataset_type": "FMRI",
        "num_atoms": 100,
        "batch_size": 1000,
        "train_mode": True,
        "train_model_B": False,
        "adjust_Psi": False,
        "clip": 5,
        "feature_size": 3,
        "epochs": 75,
        "start_updating_f_after": 300,
        "update_f_every_N_steps": 5,
        "minimize_neg_terms_until": 0,
        "downward_critics_config": {
            "hidden_sizes_v_critic": [512, 1024, 1024, 512],
            "hidden_sizes_xi_critic": [512, 512, 512],
            "critic_output_size": 32,
            "lr": 1e-3,
            "bias": True,
            "weight_decay": 0,
        },
        
        "decoupled_critic_config": {
            "hidden_sizes_encoder_1": [512, 512, 512],
            "hidden_sizes_encoder_2": [512, 512, 512],
            "critic_output_size": 32,
            "lr": 1e-3,
            "bias": True,
            "weight_decay": 0,
        },
        "feature_network_config": {
            "hidden_sizes": [256, 256, 256, 256, 256],
            "lr": 1e-4,
            "bias": True,
            "weight_decay": 1e-6,
        }
    }

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    skip_model = SkipConnectionSupervenientFeatureNetwork(
        num_atoms=config['num_atoms'],
        feature_size=config['feature_size'],
        hidden_sizes=config['feature_network_config']['hidden_sizes'],
        include_bias=config['feature_network_config']['bias'],
    ).to(device)

    project_name = "NEURIPS-FMRI-model-A"

    skip_model, name = train_feature_network(
        config=config,
        trainloader=trainloader,
        feature_network_training=skip_model,
        project_name=project_name,
        model_dir_prefix=project_name
    )

    config_test = {    
        "torch_seed": seed,
        "dataset_type": "FMRI",
        "num_atoms": 100,
        "batch_size": 1000,
        "train_mode": False,
        "train_model_B": False,
        "adjust_Psi": True,
        "clip": 5,
        "feature_size": 3,
        "epochs": 10,
        "start_updating_f_after": 300,
        "update_f_every_N_steps": 5,
        "minimize_neg_terms_until": 0,
        "downward_critics_config": {
            "hidden_sizes_v_critic": [512, 1024, 1024, 512],
            "hidden_sizes_xi_critic": [512, 512, 512],
            "critic_output_size": 32,
            "lr": 1e-3,
            "bias": True,
            "weight_decay": 0,
        },
        
        "decoupled_critic_config": {
            "hidden_sizes_encoder_1": [512, 512, 512],
            "hidden_sizes_encoder_2": [512, 512, 512],
            "critic_output_size": 32,
            "lr": 1e-3,
            "bias": True,
            "weight_decay": 0,
        },
        "feature_network_config": {
            "hidden_sizes": [256, 256, 256, 256, 256],
            "lr": 1e-4,
            "bias": True,
            "weight_decay": 1e-6,
        }
    }

    project_name_test = project_name+ "-verification"

    skil_model, _ = train_feature_network(
            config=config_test,
            trainloader=trainloader,
            feature_network_training=skip_model,
            project_name=project_name_test,
            model_dir_prefix=None
    )




