from td3.td3_agent_CarRacing import CarRacingTD3Agent
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_name", default="TD3_austria")
	parser.add_argument("--scenario", type=str, default="austria_competition")
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--noise_clip_rate", default=0.5, type=float)
	parser.add_argument("--PER", default=True, type=bool)
	args = parser.parse_args()

	# my hyperparameters, you can change it as you like
	config = {
		"gpu": True,
		"training_steps": int(1e8),
		"gamma": 0.99,
		"tau": 0.005,
		"batch_size": 32,
		"warmup_steps": 1000,
		"total_episode": int(1e7),
		"lra": 4.5e-5,
		"lrc": 4.5e-5,
		"replay_buffer_capacity": int(1e5),
		"logdir": f'/disk1/nfs/bwdong/rl_final/log/ver2/{args.exp_name}',
		"update_freq": 2,
		"eval_interval": 50,
		"eval_episode": 1,
		"noise_clip_rate": args.noise_clip_rate,
		"device": args.device,
		"PER": args.PER,
		"scenario": args.scenario,
	}

	agent = CarRacingTD3Agent(config)
	agent.train()
