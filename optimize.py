import os
import sys
import shutil
import json

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from train import train


def save_results(analysis, experiment_name: str) -> None:
    """ Save the results of the best trial to a file """

    best_trial = analysis.get_best_trial()
    result_path = os.path.join(best_trial.logdir, "progress.csv")
    config_path = os.path.join(best_trial.logdir, "params.json")

    experiment_dir = os.path.join(os.environ["HUK_CHAL"], "experiments", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    shutil.copyfile(result_path, os.path.join(experiment_dir, "progress.csv"))
    shutil.copyfile(config_path, os.path.join(experiment_dir, "params.json"))

    print("Best trial's RMSE:", best_trial.last_result["avg_rmse"])


def tune_params(config: dict, experiment_name: str) -> None:
    """ Tune the hyperparameters of the model """
    
    hpo_config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": 16,
        "momentum": tune.uniform(0.8, 0.99),
        # "weight_decay": tune.uniform(0.00001, 0.001)
    }
    config.update(hpo_config)

    reporter = CLIReporter(
        max_report_frequency=60,
        metric_columns=["avg_rmse", "rmse_hood", "rmse_backdoor_left", "train_loss", "val_loss"]
    )

    scheduler = ASHAScheduler(
        max_t=config["max_epochs"],                     # Maximum training iterations (epochs)
        reduction_factor=config["reduction_factor"]     # Reduction factor for stopping trials
    )

    analysis = tune.run(
        train,
        config=config,
        metric='avg_rmse',                              # Metric to optimize
        mode='min',                                     # Optimization direction
        num_samples=config["hpo_samples"],              # Number of trials
        scheduler=scheduler,                            
        resources_per_trial={"cpu": 4, "gpu": 1},       # Number of CPU cores and GPUs per trial
        time_budget_s=config["time_budget"],            # Time budget in seconds
        name=experiment_name,
        local_dir=os.path.join(os.environ["HUK_CHAL"], "raytune"),
        progress_reporter=reporter,
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id.split('_')[1]}"
    )

    save_results(analysis, experiment_name)
    

if __name__ == "__main__":
    with open(sys.argv[1], "r") as config:
        config = json.load(config)
    experiment_name = sys.argv[2]
    tune_params(config, experiment_name)