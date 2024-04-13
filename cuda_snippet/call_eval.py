import requests
import json

# 设置目标 URL
# url = 'http://10.166.188.17:8080'
url = 'http://xxx:8080'

# 准备发送的数据

prompt=[
    ' 你好',
        ]
# import json
# prompt=json.load(data)
thinking_budget = 2048
data = {
     # 'input': prompt[2],
    'prompt': prompt[-1],
    'temperature': 0.8,
    "thinking_budget": thinking_budget,
    "max_new_tokens": 30000,
    "top_p": 0.95,
    "top_k": 10,
    # "random_seed": 0,
    # "logprobs": 2,
}
data_params = {
    'text': json.dumps(data)
}
# 发送 POST 请求
response = requests.post(url, json=data, timeout=(60, 600))
# response = requests.get(url, params=data_params)
# 打印响应内容
response_data = response.json()
print(response_data)