import jieba
from difflib import SequenceMatcher
from collections import Counter
import re
from pypinyin import lazy_pinyin

class LightweightClauseComparator:
    def __init__(self):
        # 定义需要忽略的标点符号
        self.punctuation = '，。！？；：""''（）【】《》、'
        
        # 停用词列表
        self.stopwords = set(['的', '了', '且', '与', '和', '或', '由', '从', '到', '对', '该', '此'])
        
        # 数字标准化映射
        self.number_map = {
            '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
            '六': '6', '七': '7', '八': '8', '九': '9', '十': '10',
            '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5',
            '陆': '6', '柒': '7', '捌': '8', '玖': '9', '拾': '10'
        }

    def preprocess(self, text):
        """文本预处理"""
        # 移除空白字符
        text = re.sub(r'\s+', '', text)
        
        # 标准化数字表示
        for cn_num, arab_num in self.number_map.items():
            text = text.replace(cn_num, arab_num)
            
        # 括号标准化
        text = text.replace('（', '(').replace('）', ')')
        
        return text

    def remove_punctuation(self, text):
        """移除标点符号"""
        return ''.join(char for char in text if char not in self.punctuation)

    def get_words_similarity(self, text1, text2):
        """计算分词后的词袋相似度"""
        # 分词
        words1 = [w for w in jieba.cut(text1) if w not in self.stopwords]
        words2 = [w for w in jieba.cut(text2) if w not in self.stopwords]
        
        # 构建词频向量
        counter1 = Counter(words1)
        counter2 = Counter(words2)
        
        # 计算交集
        common_words = set(counter1.keys()) & set(counter2.keys())
        
        if not common_words:
            return 0.0
            
        # 计算词频相似度
        numerator = sum(min(counter1[word], counter2[word]) for word in common_words)
        denominator = max(sum(counter1.values()), sum(counter2.values()))
        
        return numerator / denominator

    def get_char_similarity(self, text1, text2):
        """计算字符级别相似度"""
        return SequenceMatcher(None, text1, text2).ratio()

    def get_pinyin_similarity(self, text1, text2):
        """计算拼音相似度"""
        pinyin1 = lazy_pinyin(text1)
        pinyin2 = lazy_pinyin(text2)
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
            return True, 0.99, "仅标点符号差异"
        
        # 3. 多维度相似度计算
        words_sim = self.get_words_similarity(clean_a_no_punct, clean_b_no_punct)
        char_sim = self.get_char_similarity(clean_a_no_punct, clean_b_no_punct)
        pinyin_sim = self.get_pinyin_similarity(clean_a_no_punct, clean_b_no_punct)
        
        # 加权平均，词袋相似度权重最高
        weighted_sim = (
            words_sim * 0.5 +     # 词袋相似度
            char_sim * 0.3 +      # 字符相似度
            pinyin_sim * 0.2      # 拼音相似度
        )
        
        # 计算差异详情
        details = {
            "词袋相似度": f"{words_sim:.3f}",
            "字符相似度": f"{char_sim:.3f}",
            "拼音相似度": f"{pinyin_sim:.3f}"
        }
        
        if weighted_sim >= threshold:
            return True, weighted_sim, details
        else:
            return False, weighted_sim, details

# 使用示例
def main():
    comparator = LightweightClauseComparator()
    
    # 测试用例
    test_cases = [
        # # 完全相同
        # ("甲方应在收到乙方通知后30日内付款。", "甲方应在收到乙方通知后30日内付款。"),
        # # 标点符号差异
        # ("甲方应在收到乙方通知后30日内付款。", "甲方应在收到乙方通知后30日内付款"),
        # # 数字表达方式不同
        # ("甲方应在收到乙方通知后三十日内付款。", "甲方应在收到乙方通知后30日内付款。"),
        # # 同义表达
        # ("甲方应在收到乙方通知后的三十日内，完成付款。", "甲方应当在接到乙方通知后的30天内，支付款项。"),
        # # 明显不同
        # ("甲方应在收到乙方通知后30日内付款。", "乙方应在收到甲方通知后30日内交付货物。")
        # 
        (
            """国产自主品牌
            交换容量 >= 150 Gbps包转发率 >= 60 mpps,
            扩展槽数 >=2,
            支持千兆光接口数量：>= 24个（Combo接口数量：>= 8个）,
            支持百兆千兆SFP自适应,
            支持802.1Q,VLAN数量 >=1000个,
            路由表 >= 8K,
            MAC地址表 >= 12K,
            支持IPv4静态路由、RIP V1/V2、OSPF、BGP,
            支持IPv6静态路由、RIPng、OSPFv3、BGP4+,
            支持IPv4和IPv6环境下的策略路由,
            支持出方向ACL，以便于灵活实现数据包过滤,
            支持出方向的流量限速功能,
            支持多对一的端口镜像,
            支持ARP 入侵检测功能,
            支持DHCP Snooping,
            支持IGMP Snooping,
            支持MLD，MLD Snooping等IPv6组播协议,
            支持基于第二层、第三层和第四层的ACL,
            支持基于端口和VLAN的 ACL,
            支持IPv6 ACL,
            支持多种管理方式：CLI、Telnet，Web，SNMPv1/v2/v3""",
           """国产自主品牌
            交换容量 >= 150 Gbps包转发率 >= 60 mpps,
            扩展槽数 >=2,
            支持千兆光接口数量：>= 24个（Combo接口数量：>= 8个）,
            支持百兆千兆SFP自适应,
            支持802.1Q,VLAN数量 >=1000个,
            路由表 >= 8K,
            MAC地址表 >= 12K,
            支持IPv4静态路由、RIP V1/V2、OSPF、BGP,
            支持IPv6静态路由、RIPng、OSPFv3、BGP4+,
            支持IPv4和IPv6环境下的策略路由,
            支持出方向ACL，以便于灵活实现数据包过滤,
            支持ARP 入侵检测功能,
            支持DHCP Snooping,
            支持IGMP Snooping,
            支持出方向的流量限速功能,
            支持多对一的端口镜像,
            支持MLD，MLD Snooping等IPv6组播协议,
            支持基于第二层、第三层和第四层的ACL,
            支持基于端口和VLAN的 ACL,
            支持IPv6 ACL,
            支持多种管理方式：CLI、Telnet，Web，SNMPv1/v2/v3"""
        )
    ]
    
    for i, (clause_a, clause_b) in enumerate(test_cases, 1):
        is_same, similarity, details = comparator.is_consistent(clause_a, clause_b)
        print(f"\n测试用例 {i}:")
        print(f"条款A: {clause_a}")
        print(f"条款B: {clause_b}")
        print(f"是否一致: {is_same}")
        print(f"综合相似度: {similarity:.3f}")
        print("详细指标:", details)

if __name__ == "__main__":
    main()