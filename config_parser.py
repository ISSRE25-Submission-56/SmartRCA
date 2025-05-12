import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="SmartRCA Configuration")

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to the configuration YAML file')
    parser.add_argument('--dataset', type=str, help='Override dataset name')
    parser.add_argument('--device', type=str, help='Override device setting (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, help='Override output directory')

    args = parser.parse_args()
    return args


def merge_args_with_config(args, config):
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.device:
        config['embedding']['device'] = args.device
    if args.output_dir:
        config['output_dir'] = args.output_dir
    return config


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    config = merge_args_with_config(args, config)

    print("Loaded Configuration:")
    print(yaml.dump(config, sort_keys=False))
