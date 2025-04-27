from src.entity_extraction import extract_entities
from src.embedding_clustering import cluster_embeddings
from src.knowledge_graph import build_knowledge_graph
from src.rca_inference import root_cause_analysis
from src.utils import preprocess_log

def main():
    # 示例日志
    logs = "Example log: System failure detected due to memory overflow and hardware issues."
    
    # Step 1: 日志预处理
    processed_logs = preprocess_log(logs)
    
    # Step 2: 实体抽取
    entities, noun_phrases = extract_entities(processed_logs)
    
    # Step 3: 嵌入聚类进行实体标准化
    clusters = cluster_embeddings(entities)
    
    # Step 4: 构建知识图谱
    kg = build_knowledge_graph(entities, clusters)
    
    # Step 5: 根因分析
    cause = root_cause_analysis(processed_logs, kg)
    
    print(f"Identified Root Cause: {cause}")

if __name__ == "__main__":
    main()
