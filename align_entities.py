import argparse
import os
import yaml
from utils.entity_extraction import extract_entities_from_logs
from utils.embedding_utils import embed_entities
from utils.clustering import cluster_and_normalize
from utils.io_utils import save_aligned_entities


def run_alignment(config):
    log_path = config['entity_alignment']['log_path']
    output_path = config['entity_alignment']['output_path']
    embedding_model = config['embedding']['model']
    device = config['embedding']['device']
    
    print("[1/3] Extracting raw entities from logs...")
    raw_entities = extract_entities_from_logs(log_path)

    print("[2/3] Embedding and clustering entities...")
    entity_embeddings = embed_entities(raw_entities, model_name=embedding_model, device=device)
    normalized_entities, cluster_map = cluster_and_normalize(raw_entities, entity_embeddings)

    print("[3/3] Saving aligned entities...")
    save_aligned_entities(normalized_entities, cluster_map, output_path)

    print(f"Entity alignment completed. Aligned entities saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Entity Alignment Script for SmartRCA")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    run_alignment(config)
