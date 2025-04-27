from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np

def cluster_embeddings(entities):
    """
    对抽取的实体进行嵌入并进行聚类。
    
    Args:
    entities (list): 抽取的实体列表
    
    Returns:
    list: 聚类结果，返回每个实体对应的聚类标签
    """
    # 使用SentenceTransformer将实体转换为嵌入
    model = SentenceTransformer('all-MiniLM-L6-v2')
    entity_texts = [entity['word'] for entity in entities]  # 提取实体文本
    embeddings = model.encode(entity_texts)
    
    # 使用KMeans进行聚类
    kmeans = KMeans(n_clusters=5, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    
    # 将聚类结果返回
    return clusters
