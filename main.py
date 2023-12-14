from td3_agent_CarRacing import CarRacingTD3Agent
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default="TD3_austria")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--update_freq", default=2, type=int)
	parser.add_argument("--noise_clip_rate", default=0.2, type=float)
	parser.add_argument("--PER", default=False, type=bool)
	args = parser.parse_args()

	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": int(3e6),
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": int(5e6),
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": 4096,
		"logdir": f'/disk1/nfs/bwdong/rl_final/log/{args.exp_name}',
		"update_freq": args.update_freq,
		"eval_interval": 100,
		"eval_episode": 5,
		"noise_clip_rate": args.noise_clip_rate,
		"device": args.device,
		"PER": args.PER,
	}

	agent = CarRacingTD3Agent(config)
	agent.train()