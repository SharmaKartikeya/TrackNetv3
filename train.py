import argparse
from utils.config import *
from agents import *


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    arg_parser.add_argument(
        '-model',
        '--model-config',
        default='configs/tracknetv3.yaml',
        help='The model configuration file in yaml format',
    )
    
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    run_config = 'configs/train.yaml'

    # parse the config json file
    config = process_configs([args.model_config, run_config])
    

    agent_class = globals()[config.agent]
    agent = agent_class(config)

    agent.run()


if __name__ == '__main__':
    main()