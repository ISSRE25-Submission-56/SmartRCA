import os
import json
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
import argparse
from config_parser import load_config, merge_args_with_config
from entity_extraction import extract_entities
from entity_embedding import generate_embeddings
from entity_alignment import align_entities

# Setup logger
logger = logging.getLogger('SmartRCA')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def build_knowledge_graph(config):
    # Load dataset
    dataset_path = config['dataset']['path']
    log_file = os.path.join(dataset_path, config['dataset']['log_file'])
    
    # Load logs
    with open(log_file, 'r') as f:
        logs = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(logs)} logs from {log_file}")

    # Extract entities from logs
    entity_extraction_method = config['entity_extraction']['method']
    entities = extract_entities(logs, method=entity_extraction_method)
    logger.info(f"Extracted {len(entities)} entities from logs")

    # Generate entity embeddings
    embedding_model = config['embedding']['model']
    entity_embeddings = generate_embeddings(entities, model=embedding_model)
    logger.info(f"Generated embeddings for {len(entities)} entities")

    # Perform entity alignment
    entity_alignment_file = config['alignment']['output_mapping']
    aligned_entities = align_entities(entities, entity_embeddings, config['embedding']['device'])
    logger.info(f"Aligned {len(aligned_entities)} entities")

    # Build Knowledge Graph
    kg = nx.Graph()
    
    # Add nodes (entities)
    for entity in aligned_entities:
        kg.add_node(entity)
    
    # Add edges (relations based on co-occurrence)
    for i, log in enumerate(logs):
        for j, entity1 in enumerate(log['entities']):
            for k, entity2 in enumerate(log['entities']):
                if j < k:  # Ensure each pair is added only once
                    similarity = cosine_similarity([entity_embeddings[entity1]], [entity_embeddings[entity2]])[0][0]
                    if similarity > 0.7:  # Set a threshold for similarity
                        kg.add_edge(entity1, entity2, weight=similarity)
    
    logger.info(f"Built knowledge graph with {len(kg.nodes)} nodes and {len(kg.edges)} edges")

    # Save the knowledge graph to a file
    kg_output_file = config['knowledge_graph']['kg_output_file']
    nx.write_edgelist(kg, kg_output_file)
    logger.info(f"Saved knowledge graph to {kg_output_file}")

    # Return the graph for further processing if needed
    return kg

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="SmartRCA Knowledge Graph Builder")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Merge arguments with config
    config = merge_args_with_config(args, config)

    # Build Knowledge Graph
    build_knowledge_graph(config)

if __name__ == "__main__":
    main()
