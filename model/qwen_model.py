import os

from langchain_openai import ChatOpenAI


class QwenModel:
    def __init__(self):
        self.client = ChatOpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"
        )

    def query(self, user: str, system: str) -> str:
        return self.client.invoke(
            # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        ).content
