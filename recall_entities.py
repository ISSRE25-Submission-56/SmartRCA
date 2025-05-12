import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def recall_entities(log_text, recall_results, top_k=5):
    """
    通过计算相似度从已召回的实体中选择与日志最相关的 top_k 个实体。
    
    Parameters:
    - log_text: 当前日志文本
    - recall_results: 召回的实体及其向量的字典
    - top_k: 召回的最大实体数目
    
    Returns:
    - top_k_entities: 与日志文本最相关的 top_k 个实体
    """
    
    # 将日志文本向量化（假设这里已经有一个预训练的向量模型，这里只是示例）
    log_vector = vectorize_log_text(log_text)
    
    # 从 recall_results 中提取实体及其嵌入
    entity_embeddings = recall_results.get('entity_embeddings', {})
    entities = list(entity_embeddings.keys())
    embeddings = np.array(list(entity_embeddings.values()))
    
    # 计算日志文本与每个实体的余弦相似度
    similarities = cosine_similarity([log_vector], embeddings).flatten()
    
    # 获取 top_k 个最相似的实体
    top_k_idx = np.argsort(similarities)[-top_k:][::-1]
    top_k_entities = [entities[idx] for idx in top_k_idx]
    
    return top_k_entities

def vectorize_log_text(log_text):
    """
    假设这是将日志文本向量化的函数，实际中可以使用像BERT之类的预训练模型。
    
    Parameters:
    - log_text: 输入的日志文本
    
    Returns:
    - log_vector: 向量化后的日志文本
    """
    # 这里假设直接返回一个随机生成的向量作为示例
    # 在实际应用中，您需要使用适当的预训练模型（如BERT）对日志文本进行向量化
    np.random.seed(42)
    return np.random.rand(512)

def load_recall_results(recall_output_file):
    """
    从指定的文件中加载实体召回结果。
    
    Parameters:
    - recall_output_file: 存储实体召回结果的 JSON 文件路径
    
    Returns:
    - recall_results: 加载的实体召回结果
    """
    with open(recall_output_file, 'r') as f:
        recall_results = json.load(f)
    return recall_results

def main():
    # 指定召回结果文件路径和日志文本文件路径
    recall_output_file = 'outputs/recall_results.json'
    
    # 加载召回结果
    recall_results = load_recall_results(recall_output_file)
    
    # 假设我们有一组待处理的日志
    test_logs = [
        {"id": 1, "text": "Connection timeout error during database query."},
        {"id": 2, "text": "CPU utilization exceeds threshold in server node."},
    ]
    
    # 对每个日志文本进行实体召回
    for log in test_logs:
        log_id = log['id']
        log_text = log['text']
        
        # 召回与日志文本最相关的 top_k 实体
        top_k_entities = recall_entities(log_text, recall_results, top_k=5)
        
        print(f"Log ID: {log_id}, Top-K Entities: {top_k_entities}")

if __name__ == "__main__":
    main()
