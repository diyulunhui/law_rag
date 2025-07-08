import os
import numpy as np

from typing import List, Dict, Tuple
from openai import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter


class LegalTextEmbedder:
    def __init__(self, dimensions: int = 1024, model: str = 'text-embedding-v4'):
        self.encoding_format = 'float'
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
        )
        self.dimensions = dimensions
        self.model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )
        self.small_to_big_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len
        )

    def embedding(self, text: str) -> list[float]:
        return self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,  # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
            encoding_format=self.encoding_format
        ).data[0].embedding

    def embed(self, texts: List[str], chunk_size: int = 250, chunk_overlap: int = 50) -> List[List[float]]:
        """生成文本向量，自动处理长文本"""
        # 1. 文本预处理和分块
        chunks = []
        chunk_info = []  # 记录每个chunk的原始文本索引
        for i, text in enumerate(texts):
            splits = self.text_splitter.split_text(text)
            chunks.extend(splits)
            chunk_info.extend([i] * len(splits))

        embeddings = [self.embedding(chunk) for chunk in chunks]

        # 3. 重组为原始文本对应的嵌入
        text_embeddings = [[] for _ in range(len(texts))]
        for emb, orig_idx in zip(embeddings, chunk_info):
            text_embeddings[orig_idx].append(emb)

        # 4. 对分块嵌入进行平均池化
        return [np.mean(embs, axis=0) if embs and len(embs) > 0 else [] for embs in text_embeddings]

    def small_to_big_emb(self, texts: List[str]):
        """生成文本向量，自动处理长文本"""
        # 1. 文本预处理和分块
        chunks = []
        chunk_info = []  # 记录每个chunk的原始文本索引
        for i, text in enumerate(texts):
            splits = self.small_to_big_text_splitter.split_text(text)
            chunks.extend(splits)
            chunk_info.extend([i] * len(splits))

        embeddings = [self.embedding(chunk) for chunk in chunks]

        return embeddings, chunk_info
