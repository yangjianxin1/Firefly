import json
import httpx


def main():
    url = 'http://127.0.0.1:8877/firefly'
    timeout = 60    # 超时设置

    # 生成超参数
    max_new_tokens = 500
    top_p = 0.85
    temperature = 0.35
    repetition_penalty = 1.0
    do_sample = True

    inputs = '背诵李白的将进酒'  # 请求内容
    inputs = inputs.strip()

    params = {
        "inputs": inputs,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample
    }

    timeout = httpx.Timeout(timeout)
    headers = {"Content-Type": "application/json", "Connection": "close"}
    session = httpx.Client(base_url="", headers=headers)
    response = session.request("POST", url, json=params, timeout=timeout)
    result = json.loads(response.text)['output']
    print(result)


if __name__ == '__main__':
    main()
