# -*- coding: utf-8 -*-


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from data_loader.file_loader import FileLoader
from embadder.Legal_text_embedder import LegalTextEmbedder
from model.qwen_model import QwenModel
from retrieve.retrieve import LegalRetrievalSystem, LegalCaseRetriever


def create_legal_qa_chain(retrieval_system: LegalRetrievalSystem, qwen_model: QwenModel):
    # 自定义提示模板
    prompt_template = """你是一名专业法官，请根据以下法律案例和用户问题生成回答：
    【相关案例】
    {context}

    【用户问题】
    {question}

    要求：
    1. 引用具体案例ID（如：参考案例2023民终1234号）
    2. 分析法律依据（如：根据《[民法典](@replace=10001)》第XXX条）
    3. 给出明确结论"""

    retriever = LegalCaseRetriever(retrieval_system)
    qa_chain = RetrievalQA.from_chain_type(
        llm=qwen_model.client,  # 适配OpenAI兼容接口
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        },
        return_source_documents=True
    )
    return qa_chain


# ================== 示例使用 ==================
if __name__ == '__main__':
    # 初始化系统（保持不变）
    loader = FileLoader("/raw_rag/wenshu_dataset/dev")
    embedder = LegalTextEmbedder()
    retrieval_system = LegalRetrievalSystem(embedder)

    # 尝试加载已有索引，不存在则构建
    if not retrieval_system.load_index(loader):
        retrieval_system.build_index(loader.case_list)

    # 创建LangChain问答链
    qa_chain = create_legal_qa_chain(retrieval_system, QwenModel())

    print("法律问答系统已启动，输入 'exit' 退出程序。")
    while True:
        query = input("\n请输入法律问题（或输入 'exit' 退出）: ").strip()
        if query.lower() == 'exit':
            print("感谢使用，程序已退出。")
            break

        try:
            result = qa_chain.invoke({"query": query})
            print("\n回答：", result["result"])
            print("来源案例：", [doc.metadata["case_id"] for doc in result["source_documents"]])
        except Exception as e:
            print(f"查询失败，请重试。错误信息：{e}")


