import argparse
from env import TrafficEnv, TrafficEnvMulti

def main(args): 
    if "single-intersection" in args.net:
        env = TrafficEnv(args.config, args.net, args.route, args.weights)
    else: 
        env = TrafficEnvMulti(args.config, args.net, args.route, args.weight)
        
    if args.train:
        env.train(args.path)
    else: 
        env.test(args.path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--net", type=str, required=True, help="Net file path")
    parser.add_argument("--route", type=str, required=True, help="Route file path")
    parser.add_argument("--path", type=str, required=True, help="Path to save mp4 and weights")
    parser.add_argument("--weights", type=str, required=False, help="Path to folder containing weights")
    parser.add_argument("--train", action="store_true", help="Train the model")
    
    args = parser.parse_args()
    
    main(args)