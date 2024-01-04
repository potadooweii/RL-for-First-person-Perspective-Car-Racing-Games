from PPO.PPO_car_racing_agent import AtariPPOAgent
import argparse
from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta
    parser.add_argument("--exp_name", type=str, default="PPO_austria")
    parser.add_argument(
        "--scenario", type=str, default="austria_competition_collisionStop"
    )
    parser.add_argument("--logdir", type=str, default="log/")
    # hyperparameters
    ## car racing reward
    parser.add_argument("--accumulated_progress_coeff", type=float, default=0.1)
    parser.add_argument("--velocity_coeff", type=float, default=1e-5)
    parser.add_argument("--position_coeff", type=float, default=2e-3)
    parser.add_argument("--collision_coeff", type=float, default=0.5)
    ## action
    parser.add_argument("--continuous", type=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--motor_magnitude", type=float, default=0.03)
    ## NN
    parser.add_argument("--backbone", type=Optional[str], default=None)
    ## training
    parser.add_argument("--training_steps", type=int, default=int(1e6))
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--max_gradient_norm", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=int(128))
    ### PPO only
    parser.add_argument("--update_sample_count", type=int, default=int(1e4))
    parser.add_argument("--discount_factor_gamma", type=float, default=0.99)
    parser.add_argument("--discount_factor_lambda", type=float, default=0.95)
    parser.add_argument("--update_ppo_epoch", type=int, default=int(3))
    parser.add_argument("--value_coefficient", type=float, default=0.5)
    parser.add_argument("--entropy_coefficient", type=float, default=0.01)
    parser.add_argument("--horizon", type=int, default=int(128))
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    # evaluation
    parser.add_argument("--eval_interval", type=int, default=int(100))
    parser.add_argument("--eval_episode", type=int, default=int(3))
    # hardware
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    agent = AtariPPOAgent(vars(args))
    agent.train()
