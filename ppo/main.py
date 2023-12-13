from ppo_agent_atari import AtariPPOAgent
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default="PPO_austria")
	parser.add_argument("--device", default="cuda")
	args = parser.parse_args()

	config = {
		"gpu": True,
		"training_steps": 1e8,
		"update_sample_count": 10000,
		"discount_factor_gamma": 0.99,
		"discount_factor_lambda": 0.95,
		"clip_epsilon": 0.2,
		"max_gradient_norm": 0.5,
		"batch_size": 128,
		"logdir": f'/disk1/nfs/bwdong/rl_final/log/{args.exp_name}',
		"update_ppo_epoch": 3,
		"learning_rate": 2.5e-4,
		"value_coefficient": 0.5,
		"entropy_coefficient": 0.01,
		"horizon": 128,
		"eval_interval": 100,
		"eval_episode": 5,
	}

	agent = AtariPPOAgent(config)
	agent.train()

