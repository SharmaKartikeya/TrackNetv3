import os
from datetime import datetime
import yaml
from utils.dirs import create_dirs
from utils.dictionary import ConfigDict

def get_config_from_yaml(yaml_file) -> ConfigDict:
    """
    Get the config from a yaml file
    :param yaml_file: the path of the config file
    :return: config(dictionary)
    """

    # parse the configurations from the config yaml file provided
    with open(yaml_file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            converted_dict = ConfigDict(config)
            return converted_dict
        
        except yaml.YAMLError as exc:
            print(exc)

def process_configs(yaml_files: list):
    config = ConfigDict({})
    
    for yaml_file in yaml_files:
        config.update(get_config_from_yaml(yaml_file))
    
    if config.train_size:
        config.save_path = os.path.join(
            config.save_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        print(f"Save Path: {config.save_path}")
        create_dirs([config.save_path])

    elif config.save_path:
        f = open(config.save_path, 'wb')
        f.close()
        

    return config