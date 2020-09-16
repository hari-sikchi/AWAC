
from AWAC.awac import AWAC
import argparse
import gym

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="AWAC")
    parser.add_argument("--env", default="hopper-random-v0")
    parser.add_argument("--exp_name", default="data/dump")
    parser.add_argument("--num_expert_trajs", default=5, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    env_fn = lambda:gym.make(args.env)


    if 'AWAC' in args.algorithm:
        agent = AWAC(env_fn, logger_kwargs={'output_dir':args.exp_name+'_s'+str(args.seed), 'exp_name':args.exp_name}, batch_size=1024,  seed=args.seed, algo=args.algorithm)
    else:
        raise NotImplementedError

    agent.populate_replay_buffer()
    agent.run()



