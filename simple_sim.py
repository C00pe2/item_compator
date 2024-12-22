import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import jieba
from scipy.spatial.distance import cosine

# TF-IDF + 余弦相似度

# 优点：考虑词频和重要性权重
# 适用：当词序不太重要，更关注语义相似性时
# 特点：对于长文本效果更好


# 序列相似度（编辑距离）

# 优点：考虑文本的顺序信息
# 适用：当需要严格比较文本结构时
# 特点：对标点符号和细微差异敏感


# Jaccard相似度

# 优点：计算简单直观
# 适用：快速对比文本相似程度
# 特点：只考虑词的共现情况，不考虑顺序和频率

def calculate_similarities(text1, text2):
    """
    计算两段文本之间的多种相似度指标
    
    Parameters:
    text1, text2: 待比较的两段文本
    
    Returns:
    dict: 包含多种相似度指标的字典
    """
    # 对文本进行分词
    text1_seg = ' '.join(jieba.cut(text1))
    text2_seg = ' '.join(jieba.cut(text2))
    
    # 1. TF-IDF + 余弦相似度
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform([text1_seg, text2_seg])
    cosine_sim = (tfidf_matrix * tfidf_matrix.T).toarray()[0,1]
    
    # 2. 编辑距离相似度(基于SequenceMatcher)
    sequence_sim = SequenceMatcher(None, text1, text2).ratio()
    
    # 3. Jaccard相似度
    set1 = set(jieba.cut(text1))
    set2 = set(jieba.cut(text2))
    jaccard = len(set1 & set2) / len(set1 | set2)
    
    return {
        'cosine_similarity': cosine_sim,
        'sequence_similarity': sequence_sim,
        'jaccard_similarity': jaccard
    }

# 示例使用
text1 = "甲方应当在合同签订后的十个工作日内支付预付款"
text2 = "甲方应当在合同签订后的十个工作日内支付定金"

results = calculate_similarities(text1, text2)
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")