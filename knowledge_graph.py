import networkx as nx

def build_knowledge_graph(entities, clusters):
    """
    基于实体和聚类结果构建知识图谱。
    
    Args:
    entities (list): 抽取的实体列表
    clusters (list): 聚类标签
    
    Returns:
    NetworkX Graph: 知识图谱，图的节点是实体，边是基于实体的相似度连接
    """
    G = nx.Graph()
    
    # 添加实体节点，并将聚类信息作为节点属性
    for idx, entity in enumerate(entities):
        G.add_node(entity['word'], cluster=clusters[idx])
    
    for i in range(len(entities) - 1):
        if clusters[i] == clusters[i + 1]:
            G.add_edge(entities[i]['word'], entities[i + 1]['word'])
    
    return G
