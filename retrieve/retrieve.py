import json
import os
import random
import numpy as np
import faiss
from typing import List, Tuple

import ast

from pydantic import Field

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from data_loader.file_loader import FileLoader
from embadder.Legal_text_embedder import LegalTextEmbedder
from entity.legal_case import LegalCase
from model.qwen_model import QwenModel


class LegalRetrievalSystem:
    key_word_index: dict

    def __init__(self, embedder: LegalTextEmbedder, persist_dir: str = "./faiss_index"):
        self.embedder = embedder
        self.index = None
        self.case_data = []
        self.key_word_index = {}
        self.key_word_text = {}
        self.persist_dir = persist_dir
        self.small_to_big_index = None

    def save_index(self):
        """保存所有索引到磁盘"""
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)  # 创建目录
        # 主索引
        faiss.write_index(self.index, os.path.join(self.persist_dir, "main.index"))
        # 关键词索引
        for key, idx in self.key_word_index.items():
            safe_key = "".join(c for c in key if c.isalnum())  # 清理非法字符
            faiss.write_index(idx, os.path.join(self.persist_dir, f"kw_{safe_key}.index"))
        # Small-to-Big索引
        if self.small_to_big_index:
            faiss.write_index(self.small_to_big_index, os.path.join(self.persist_dir, "stb.index"))
        # 保存案例元数据
        with open(os.path.join(self.persist_dir, "meta.json"), "w") as f:
            json.dump([case.__dict__ for case in self.case_data], f)

    def load_index(self, loader: FileLoader) -> bool:
        """从磁盘加载索引"""
        try:
            # 主索引
            self.index = faiss.read_index(os.path.join(self.persist_dir, "main.index"))
            # 关键词索引
            for fname in os.listdir(self.persist_dir):
                if fname.startswith("kw_") and fname.endswith(".index"):
                    key = fname[3:-6]
                    self.key_word_index[key] = faiss.read_index(os.path.join(self.persist_dir, fname))
            # Small-to-Big索引
            stb_path = os.path.join(self.persist_dir, "stb.index")
            if os.path.exists(stb_path):
                self.small_to_big_index = faiss.read_index(stb_path)
            for idx, case in enumerate(loader.case_list):
                for key_word in case.keywords:
                    self.key_word_text.setdefault(key_word, [])
                    self.key_word_text.get(key_word).append(case)
            # 加载案例元数据
            with open(os.path.join(self.persist_dir, "meta.json"), "r") as f:
                self.case_data = [LegalCase(**data) for data in json.load(f)]
            return True
        except Exception as e:
            print(f"加载索引失败: {e}")
            return False

    def build_index(self, cases: List[LegalCase], force_rebuild: bool = False):
        print("====开始构建索引======")
        """构建FAISS索引和分类索引"""
        self.case_data = cases

        # 1. 为每个案例生成多维度文本表示
        print("====为每个案例生成多维度文本表示======")
        texts = []
        for case in cases:
            # 组合案件关键信息作为检索文本
            case_text = f"""
                   案件类型：{case.case_type}，审理程序：{case.case_proc}，案由：{case.judge_accusation}，裁判理由：{case.judge_reason}，关键词：{', '.join(case.keywords)}，分类：{case.category.get('cat_1', '')}/{case.category.get('cat_2', '')}
                   """
            texts.append(case_text)
            print("====为每个案例生成多维度文本表示======" + case.case_id)

        # 2. 生成嵌入向量
        print("====生成嵌入向量======")
        embeddings = self.embedder.embed(texts)
        valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) > 0]
        valid_embeddings = [embeddings[i] for i in valid_indices]

        # 3. 构建FAISS索引
        print("====构建FAISS索引======")
        dimension = len(valid_embeddings[0])
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(valid_embeddings))

        # 4. 构建分类索引
        print("====构建分类索引======")
        for idx, case in enumerate(cases):

            for key_word in case.keywords:
                self.key_word_text.setdefault(key_word, [])
                self.key_word_text.get(key_word).append(case)

        for key, case_list in self.key_word_text.items():
            # 组合案件关键信息作为检索文本
            case_text = []
            for case in case_list:
                case_text.append(f"""
                    案件类型：{case.case_type}，审理程序：{case.case_proc}，案由：{case.judge_accusation}，裁判理由：{case.judge_reason}，关键词：{', '.join(case.keywords)}，分类：{case.category.get('cat_1', '')}/{case.category.get('cat_2', '')}
                                          """)
            embeddings = self.embedder.embed(case_text)
            valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) > 0]
            valid_embeddings = [embeddings[i] for i in valid_indices]
            dimension = len(valid_embeddings[0])
            self.key_word_index.setdefault(key, faiss.IndexFlatIP(dimension))
            self.key_word_index.get(key).add(np.array(valid_embeddings))

        # 5. 构建small_to_big索引
        print("====构建small_to_big索引======")
        texts = []
        for case in cases:
            # 组合案件关键信息作为检索文本
            case_text = f"""
                           案件类型：{case.case_type}，审理程序：{case.case_proc}，案由：{case.judge_accusation}，裁判理由：{case.judge_reason}，关键词：{', '.join(case.keywords)}，分类：{case.category.get('cat_1', '')}/{case.category.get('cat_2', '')}
                           """
            # 通过大模型生成摘要
            model = QwenModel()
            result = model.query(
                "请根据以下裁判文书生成200字左右的摘要，需包含案件类型、核心争议和判决结果三要素", case_text)
            texts.append(result)
        embeddings = self.embedder.embed(texts)
        valid_indices = [i for i, emb in enumerate(embeddings) if len(emb) > 0]
        valid_embeddings = [embeddings[i] for i in valid_indices]
        dimension = len(valid_embeddings[0])
        self.small_to_big_index = faiss.IndexFlatIP(dimension)
        self.small_to_big_index.add(np.array(valid_embeddings))
        """增加持久化判断逻辑"""
        if not force_rebuild and self.load_index():
            print("从缓存加载索引成功")
            return
        self.save_index()  # 构建完成后自动保存

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[LegalCase, float]]:
        """语义向量检索"""
        query_embedding = self.embedder.embed([query])[0]
        if len(query_embedding) == 0:
            return []

        # FAISS搜索
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return [(self.case_data[idx], float(score)) for idx, score in zip(indices[0], distances[0])]

    def small_to_big_semantic_search(self, query: str, k: int = 5) -> List[Tuple[LegalCase, float]]:
        """语义向量检索"""
        query_embedding = self.embedder.embed([query])[0]
        if len(query_embedding) == 0:
            return []

        # FAISS搜索
        distances, indices = self.small_to_big_index.search(np.array([query_embedding]), k)
        return [(self.case_data[int(idx)], float(score)) for idx, score in zip(indices[0], distances[0])]

    def keyword_filtered_search(self, query: str, k: int = 5) -> List[Tuple[LegalCase, float]]:
        """分类过滤后的语义检索"""

        # 1. 获取分类下的案例索引
        model = QwenModel()
        result = model.query(
            "你现在是一个法律专家,现在有这些关键字{'同业拆借', '合伙企业', '借婚姻索取财物', '婚姻', '过失', '遗嘱继承', '增值税发票', '兑现', '民事行为能力', '不动产', '标的物', '保', '雇主责任', '集资', '交', '个人经营', '债务人', '工伤', '伪造', '解除婚姻关系', '监护权', '土地证', '继承', '建设工程合同', '买卖合同', '无民事行为能力人', '民事责任', '请求权', '认定有效', '变更', '垫付', '格式合同', '行纪合同', '刑事拘留', '保险费', '法定期限', '交通事故', '抚养能力', '合同书', '合作协议', '理赔', '合同生效', '发现权', '公', '商标专用权', '补充协议', '利害关系人', '搬迁合同', '拘役', '名誉权', '被保险人', '房屋所有权', '追诉', '滞纳金', '原始取得', '保险责任', '优先受偿权', '和解', '侵权产品', '共同债务', '非法占有', '投资基金', '按份共有', '合同诈骗', '强制措施', '对价', '不完全履行', '票', '保证金', '准用', '未成年人', '要约', '恢复名誉', '银行贷款合同', '拆迁安置', '保险利益', '监事会', '融资租赁合同', '相邻权', '人身损害', '招标', '加工合同', '误工损失', '房屋产权登记', '保证保险', '标的额', '虚假宣传', '房地产开发公司', '劳动合同', '土地承包经营权', '撤回起诉', '公有', '全额赔偿', '保险金额', '变卖', '土地租赁', '管辖', '离婚协议书', '诚实信用原则', '合同履行', '担保合同', '单务合同', '公司债', '利息', '驳回起诉', '违法所得', '委托代理合同', '基金', '挂靠经营', '夫妻共同财产', '航空运输', '最高额抵押', '国有土地使用权', '抚养费', '限制民事行为能力人', '民事权利能力', '实际履行', '房屋抵押贷款', '残疾赔偿金', '房屋拆迁安置协议', '约定期限', '法定继承人', '股东会', '给付', '扣押', '情势变更', '加工承揽合同', '劳务合同', '智力成果', '土地权属争议', '书面合同', '独创性', '保险人', '强制保险', '诉讼标的', '物业管理', '承揽合同', '意思表示真实', '贷款', '承租人', '注意义务', '合同无效', '保人', '房屋产权', '代位求偿', '股份有限公司', '土地登记', '诚实信用', '胁迫', '合伙财产', '建设用地', '共同财产', '消费者权益保护', '中介合同', '家庭财产', '排除妨碍', '委托代理', '抵', '房地产开发企业', '债权转让', '共', '有偿使用', '承包经营', '惩罚性赔偿', '个体工商户', '破产管理人', '房屋共有', '精神损害赔偿', '承运人', '继续履行', '法定监护人', '公证遗嘱', '董事', '分期付款买卖', '股权转让', '夫妻关系', '放弃继承权', '履行地', '主要条款', '部分履行', '婚姻自由', '劳动仲裁', '约定解除', '共有', '劳动报酬', '人寿保险', '公司解散', '经济犯罪', '亲子鉴定', '合同解除', '不予受理', '利润分配', '利害关系', '民事主体资格', '合伙', '拆迁补偿协议', '商品房预售', '转让合同', '合并审理', '收养关系', '抗辩权', '信用卡透支', '代位求偿权', '产权变更', '全面履行', '非婚生子女', '直接利害关系人', '违约金', '缺席判决', '连带保证责任', '医疗保险', '继承人', '交付', '催告', '没收', '船员劳务合同', '婚生子女', '涉外民商事案件', '口头合同', '居间人', '保证期间', '财产分割', '监护', '人身权利', '合同约定', '共同继承', '误工费', '承包合同', '人身保', '个人财产', '土地整理', '协议无效', '传唤', '民事行为', '合伙债务', '房屋租赁', '无效合同', '人身伤害', '家庭经营', '返还财产', '承接', '企业法人', '免责条款', '承包金', '受益人', '承付', '授权', '公积金', '本案争议', '连带责任', '著作权', '驳', '赔偿损失', '监护人', '折价款', '抚养', '自然资源', '保险代理', '除斥期间', '合同的解除', '重新鉴定', '追认', '执行和解', '合', '抚养费的承担', '商品房购销合同', '劳动保险', '担保物权', '婚前财产', '继承开始', '购销合同', '公共利益', '非法侵占', '管制', '事实婚姻', '继承顺序', '保险理赔', '解除合同', '清算', '专', '混同', '平等自愿', '土地确权', '宅基地使用权', '代理权', '不法行为', '遗赠扶养协议', '法定继承', '土地买卖', '股权', '权限范围', '继承权', '离婚', '离', '无效协议', '律师代理费', '合同有效', '留置送达', '不当得利', '实际赔偿', '养老金', '合同成立', '财产保险', '存单', '债权人', '土地征收', '劳动合', '赊销', '保险合同', '租赁', '假冒', '交通事故损害赔偿', '转包', '追加被告', '担保', '抵押', '非法占地', '迟延履行', '赔偿义务', '合同的相对性', '借贷合同', '土地征用', '建设工程', '分公司', '折旧', '消除危险', '担保期', '宅基地', '返还责任', '还款协议', '保险责任范围', '财务管理', '林地使用权', '抵偿', '著作', '票据权利', '无因管理', '实际损失', '收益权', '赔偿协议', '善意取得', '保证', '无权处分', '劳务纠纷', '家庭暴力', '荣誉权', '签章', '供应合同', '合同变', '经营许可证', '保险单', '合同', '案件受理费', '主要责任', '破产申请', '冻结', '转租', '租', '附合', '劳务报酬', '托运人', '企业改', '房屋预售', '传票', '彩礼', '财产损害赔偿', '股', '返还', '预付款', '合伙协议', '赡养', '房产证', '融资', '邻接权', '传销', '集体土地所有权', '合同的履行', '增值税', '职务行为', '房屋征收', '消除影响', '保险期限', '求偿权', '商品房预售合同', '财产权', '赔礼道歉', '离婚条件', '婚姻登记', '出卖人', '出租人的义务', '合同责任', '保管', '股权变更', '保证合同', '间接损失', '委托代理人', '劳动争议', '查封', '房屋拆迁', '所有权', '拍卖合同', '出租', '经营权', '违约责任', '说明义务', '第三者责任险', '汇票', '人身损害赔偿', '搁置物', '离婚协议', '双倍赔偿', '口头委托', '合资建房', '继父母', '融资租赁', '逾期违约金', '清偿', '变更登记', '精神损失费', '处分行为', '恢复原状', '行政合同', '宅基', '占有人', '拆', '过错责任', '按揭', '保险赔偿金', '实践合同', '赔偿金', '不履行', '相邻关系', '公证', '补救措施', '出租人', '鉴定人', '承兑', '共同行为', '共同担保', '诉讼标的额', '鉴定', '复保险', '驳回', '土地使用权', '贷款人', '精神损害', '直接损失', '车船使用税', '车辆损失险', '履行抗辩权', '违法行为', '劳动合同的解除', '第三人', '复制权', '抵押权登记', '反诉', '房屋产', '股份', '兼并', '侵权行为', '董事会', '挪用公款', '监管责任', '婚', '夫妻共同财', '旁系血亲', '处分权', '债权', '定金罚则', '法定代理人', '民间借贷', '交付货物', '继子女', '鉴证', '车辆买卖合同', '一般保证', '工程质量', '股份转让', '婚约', '经营范围', '房屋过户', '赠与合同', '企业改制', '合同的成立', '票据', '预期利益', '保险事故', '赔偿数额', '风险抵押金', '质权', '破产债务人', '按揭贷款', '反担保', '承诺', '动产', '孳息', '损害赔偿', '利率', '代理人', '婚姻基础', '投资', '合法财产', '所有', '不可抗力', '委托加工合同', '确权', '和解协议', '程序合法', '借款合同', '托运单', '代理', '租金', '表见代理', '夫妻感情破裂', '农村承包经营户', '分摊', '破产清算', '拟制血亲', '关税', '法定解除', '票据权', '电视作品', '财产关系', '挂靠', '处分', '罚金', '合同履行地', '房屋权属', '书面形式', '居间合同', '法定代表人', '拍卖', '扶养', '不法侵害', '雇主责任险', '有效合同', '部分不能', '房屋拆', '合理注意义务', '追偿', '赔偿责任', '注册商标', '土地转让', '居间行为', '恶意透支', '雇佣关系', '产权登记', '公司登记', '定金', '从合同', '继受取得', '房屋赠与', '监事', '法定孳息', '质押', '委托合同', '欺诈', '保险赔偿', '从权利', '房屋买卖', '债', '合同的效力', '质权人', '抵押借款合同', '定作合同', '商业贿赂', '自诉', '保险标的', '抵押权', '共同共有', '拆迁', '拆借合同', '遗产分割', '强制性规定', '保险期间', '转载'}" + "请你从中找到最匹配用户输入的3个关键字并以如下格式输出:['A','B','C']，请严格按照格式输出",
            query)

        array = ast.literal_eval(result)
        key_word = random.choice(array)
        cat_indices = self.key_word_index[key_word]

        # 2. 在子集中进行搜索
        query_embedding = self.embedder.embed([query])[0]
        if len(query_embedding) == 0:
            return []
        distances, indices = cat_indices.search(np.array([query_embedding]), k)
        return [(self.key_word_text[key_word][int(idx)], float(score)) for idx, score in zip(indices[0], distances[0])]

    def query_expansion(self, query: str) -> str:
        """问题扩展 - 生成相关查询变体"""
        model = QwenModel()
        result = model.query(
            "你现在是一个法律专家,请根据用户的输入生成更专业的提问,请你不要生成任何与提问无关的文字,生成的专业提问是用来给大模型做rag召回的", query)
        return result

    def build_hypothetical_questions(self, case: LegalCase) -> List[str]:
        """构建假设性问题 - 为案例生成可能的问题形式"""
        questions = [
            f"关于{case.keywords[0]}的法律案例有哪些？",
            f"如何判决{case.judge_accusation.split('。')[0]}的案件？",
            f"{case.category.get('cat_1', '')}纠纷的典型案例",
            f"{case.parties[0]['Name']}与{case.parties[1]['Name']}之间的法律纠纷"
        ]
        return questions

    def retrieve_cases(self, query: str, k: int = 5) -> List[LegalCase]:
        """多路召回与重排序"""
        # 1. 多路召回 - 获取初始候选集
        semantic_results = self.semantic_search(query, k)
        keyword_results = self.keyword_filtered_search(query, k)
        stb_results = self.small_to_big_semantic_search(query, k)

        # 2. 结果去重与合并
        unique_cases = {}
        for results in [semantic_results, keyword_results, stb_results]:
            for case, score in results:
                if case.case_id not in unique_cases:
                    unique_cases[case.case_id] = (case, [score])

        # 3. 分数归一化处理
        candidates = []
        for case_id, (case, scores) in unique_cases.items():
            # 对不同召回路径的分数进行加权平均
            normalized_score = np.mean(scores) * 0.5 + max(scores) * 0.5
            case.retrieval_score = normalized_score
            candidates.append((case, normalized_score))

        # 4. 初步按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [case for case, _ in candidates[:k * 2]]  # 保留前2k个候选

        # 5. 交叉编码器重排序（使用更强大的语义模型）
        if len(top_candidates) > 0:
            reranked_cases = self._cross_encoder_rerank(query, top_candidates)
            return reranked_cases[:k]  # 返回最终top-k

        return []

    def _cross_encoder_rerank(self, query: str, candidates: List[LegalCase]) -> List[LegalCase]:
        """使用交叉编码器进行精细重排序"""
        # 1. 准备重排序输入对（查询-文档对）
        model = QwenModel()
        pairs = []
        for case in candidates:
            case_text = f"""
            案件ID：{case.case_id}
            案件类型：{case.case_type}
            案由：{case.judge_accusation}
            裁判理由：{case.judge_reason}
            判决结果：{case.judge_result}
            """
            pairs.append((query, case_text))

        # 2. 批量计算相关性分数（使用更强大的语义模型）
        scores = []
        for query_text, case_text in pairs:
            # 使用大模型评估相关性（0-1分）
            response = model.query(
                "请评估以下查询与法律案例的相关性，给出0-1的分数，只需返回数字：\n"
                f"查询：{query_text}\n"
                f"案例：{case_text}",
                "你是一个法律相关性评估专家"
            )
            try:
                score = float(response.strip())
            except:
                score = 0.5  # 默认分数
            scores.append(score)

        # 3. 组合分数并排序（结合初始检索分数和重排序分数）
        reranked_cases = []
        for idx, case in enumerate(candidates):
            # 组合分数策略：70%重排序分数 + 30%初始检索分数
            combined_score = 0.7 * scores[idx] + 0.3 * case.retrieval_score
            case.retrieval_score = combined_score
            reranked_cases.append((case, combined_score))

        # 4. 按最终分数排序
        reranked_cases.sort(key=lambda x: x[1], reverse=True)
        return [case for case, _ in reranked_cases]


class LegalCaseRetriever(BaseRetriever):
    retrieval_system: LegalRetrievalSystem = Field(..., description="Legal retrieval system instance")

    def __init__(self, retrieval_system: LegalRetrievalSystem):
        super().__init__(retrieval_system=retrieval_system)  # 通过父类初始化传递参数
        self.retrieval_system = retrieval_system

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        cases = self.retrieval_system.retrieve_cases(query)
        return [
            Document(
                page_content=f"案件ID：{case.case_id}\n案由：{case.judge_accusation}\n裁判理由：{case.judge_reason}",
                metadata={"score": case.retrieval_score, **case.__dict__}
            ) for case in cases
        ]
