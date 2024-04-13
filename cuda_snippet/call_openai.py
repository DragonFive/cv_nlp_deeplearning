from openai import OpenAI
# copy from https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_completion_client.py
# chat : https://github.com/vllm-project/vllm/blob/main/examples/online_serving/openai_chat_completion_client.py

def chat():
    # 初始化客户端，base_url 指向目标服务器
    client = OpenAI(
        base_url="http://xxx/v1",
        api_key="dummy"  # 随便填，但不能为空
    )

    try:
        # 发送请求
        response = client.completions.create(
            model="",
            prompt="你好",
            max_tokens=1,
            temperature=0.7,
            stream=False
        )
        # 打印结果
        print(response)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    chat()