from ppo.ppo_agent_atari import AtariPPOAgent
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default="PPO_austria")
	parser.add_argument("--scenario", type=str, default="austria_competition")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--continuous", default=True, type=bool)
	args = parser.parse_args()

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 4096, # 100_000
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": f'/disk1/nfs/bwdong/rl_final/log/ver2/{args.exp_name}',
		"update_ppo_epoch": 6, # 3
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"eval_interval": 30,
		"eval_episode": 1,
		"continuous": args.continuous,
		"scenario": args.scenario,
	}

	agent = AtariPPOAgent(config)
	agent.train()

