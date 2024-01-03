from TD3.TD3_car_racing_agent import CarRacingTD3Agent
import argparse
from typing import Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # meta
    parser.add_argument("--exp_name", type=str, default="TD3_austria")
    parser.add_argument("--scenario", type=str, default="austria_competition_collisionStop")
    parser.add_argument("--logdir", type=str, default="log/")
    # hyperparameters
    ## car racing reward
    parser.add_argument("--accumulated_progress_coeff", type=float, default=0.1)
    parser.add_argument("--velocity_coeff", type=float, default=1e-5)
    parser.add_argument("--position_coeff", type=float, default=2e-3)
    parser.add_argument("--collision_coeff", type=float, default=0.5)
    ## action
    parser.add_argument("--motor_magnitude", type=float, default=0.03)
    ## NN
    parser.add_argument("--backbone", type=Optional[str], default=None)
    ## training
    parser.add_argument("--training_steps", type=int, default=int(1e6))
    parser.add_argument("--total_episode", type=int, default=int(1e7))
    parser.add_argument("--lra", type=float, default=4.5e-4)
    parser.add_argument("--lrc", type=float, default=4.5e-4)
    parser.add_argument("--batch_size", type=int, default=int(32))
    ### TD3 only
    parser.add_argument("--discount_factor_gamma", type=float, default=0.99)
    parser.add_argument("--soft_update_tau", type=float, default=0.005)
    parser.add_argument("--warmup_steps", type=int, default=int(1e3))
    parser.add_argument("--replay_buffer_capacity", type=int, default=int(1e5))
    parser.add_argument("--update_freq", type=int, default=int(2))
    parser.add_argument("--noise_clip_rate", default=0.5, type=float)
    parser.add_argument("--PER", type=argparse.BooleanOptionalAction, default=True)
    # evaluation
    parser.add_argument("--eval_interval", type=int, default=int(50))
    parser.add_argument("--eval_episode", type=int, default=int(10))
    # hardware
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    agent = CarRacingTD3Agent(vars(args))
    agent.train()
