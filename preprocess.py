import os
import json
import logging
import argparse
from config_parser import load_config, merge_args_with_config
from entity_extraction import extract_entities
from entity_embedding import generate_embeddings
from sklearn.model_selection import train_test_split

# Setup logger
logger = logging.getLogger('SmartRCA')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def preprocess_logs(config):
    # Load dataset
    dataset_path = config['dataset']['path']
    log_file = os.path.join(dataset_path, config['dataset']['log_file'])
    label_file = os.path.join(dataset_path, config['dataset']['label_file'])

    # Load logs
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(logs)} logs from {log_file}")

    # Load labels
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    logger.info(f"Loaded labels from {label_file}")

    # Extract entities from logs
    entity_extraction_method = config['entity_extraction']['method']
    entities = extract_entities(logs, method=entity_extraction_method)
    logger.info(f"Extracted {len(entities)} entities from logs")

    # Generate embeddings for the entities
    embedding_model = config['embedding']['model']
    entity_embeddings = generate_embeddings(entities, model=embedding_model)
    logger.info(f"Generated embeddings for {len(entities)} entities")

    # Split the dataset into training and testing sets (if needed)
    test_size = 0.2  # You can adjust the test size here
    train_logs, test_logs = train_test_split(logs, test_size=test_size, random_state=config['random_seed'])
    
    logger.info(f"Split the dataset into {len(train_logs)} training logs and {len(test_logs)} testing logs")

    # Save preprocessed logs and labels
    preprocessed_train_file = os.path.join(config['output_dir'], 'train_logs.jsonl')
    preprocessed_test_file = os.path.join(config['output_dir'], 'test_logs.jsonl')
    
    with open(preprocessed_train_file, 'w') as f:
        for log in train_logs:
            f.write(json.dumps(log) + '\n')
    
    with open(preprocessed_test_file, 'w') as f:
        for log in test_logs:
            f.write(json.dumps(log) + '\n')

    logger.info(f"Saved preprocessed training logs to {preprocessed_train_file}")
    logger.info(f"Saved preprocessed testing logs to {preprocessed_test_file}")

    # Optionally save embeddings
    embeddings_file = os.path.join(config['output_dir'], 'entity_embeddings.npy')
    np.save(embeddings_file, entity_embeddings)
    logger.info(f"Saved entity embeddings to {embeddings_file}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="SmartRCA Preprocessing")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Merge arguments with config
    config = merge_args_with_config(args, config)

    # Perform preprocessing
    preprocess_logs(config)

if __name__ == "__main__":
    main()
