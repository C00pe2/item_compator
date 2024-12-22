import jieba
from sentence_transformers import SentenceTransformer
import numpy as np
from difflib import SequenceMatcher
import re
from pypinyin import lazy_pinyin

class ClauseComparator:
    def __init__(self):
        # 加载中文BERT模型
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 定义需要忽略的标点符号
        self.punctuation = '，。！？；：""''（）【】《》、'
        
        # 停用词列表
        self.stopwords = set(['的', '了', '且', '与', '和', '或', '由', '从', '到', '对', '该', '此'])

    def preprocess(self, text):
        """预处理文本"""
        # 统一全角转半角、简体转繁体等
        text = text.strip()
        text = ''.join(char.strip() for char in text if char.strip())
        # 移除多余空白字符
        text = re.sub(r'\s+', '', text)
        return text

    def remove_punctuation(self, text):
        """移除标点符号"""
        return ''.join(char for char in text if char not in self.punctuation)

    def get_semantic_similarity(self, text1, text2):
        """计算语义相似度"""
        # 使用BERT模型获取文本嵌入
        embeddings = self.model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity

    def get_char_similarity(self, text1, text2):
        """计算字符级别相似度"""
        return SequenceMatcher(None, text1, text2).ratio()

    def get_pinyin_similarity(self, text1, text2):
        """计算拼音相似度，处理同音字"""
        pinyin1 = ' '.join(lazy_pinyin(text1))
        pinyin2 = ' '.join(lazy_pinyin(text2))
        return SequenceMatcher(None, pinyin1, pinyin2).ratio()

    def is_consistent(self, clause_a, clause_b, threshold=0.9):
        """判断两个条款是否一致"""
        # 预处理
        clean_a = self.preprocess(clause_a)
        clean_b = self.preprocess(clause_b)
        
        # 1. 完全相同判断
        if clean_a == clean_b:
            return True, 1.0, "完全匹配"
            
        # 2. 去除标点符号后判断
        clean_a_no_punct = self.remove_punctuation(clean_a)
        clean_b_no_punct = self.remove_punctuation(clean_b)
        if clean_a_no_punct == clean_b_no_punct:
            return True, 0.99, "标点符号差异"
            
        # 3. 计算多个维度的相似度
        semantic_sim = self.get_semantic_similarity(clean_a, clean_b)
        char_sim = self.get_char_similarity(clean_a_no_punct, clean_b_no_punct)
        pinyin_sim = self.get_pinyin_similarity(clean_a_no_punct, clean_b_no_punct)
        
        # 加权平均
        weighted_sim = (
            semantic_sim * 0.5 +  # 语义相似度权重最高
            char_sim * 0.3 +     # 字符相似度次之
            pinyin_sim * 0.2     # 拼音相似度权重最低
        )
        
        # 返回结果
        if weighted_sim >= threshold:
            reason = f"综合相似度: {weighted_sim:.3f}"
            return True, weighted_sim, reason
        else:
            reason = f"相似度不足: {weighted_sim:.3f}"
            return False, weighted_sim, reason

# 使用示例
def main():
    comparator = ClauseComparator()
    
    # 示例条款
    clause_a = "甲方应当在收到乙方书面通知后的三十（30）日内，对乙方完成的工作成果进行验收，并出具书面验收报告。"
    clause_b = "甲方应当在收到乙方书面通知后的三十日内，对乙方完成的工作成果进行验收并出具书面验收报告。"
    
    is_same, similarity, reason = comparator.is_consistent(clause_a, clause_b)
    print(f"条款一致性: {is_same}")
    print(f"相似度得分: {similarity:.3f}")
    print(f"判断依据: {reason}")

if __name__ == "__main__":
    main()