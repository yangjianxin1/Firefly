import json


def firefly2chatglm3():
    """
    将firefly的训练数据格式，转换成chatglm3格式
    """
    file = '../data/dummy_data.jsonl'
    save_file = '../data/dummy_data_chatglm3.jsonl'
    with open(file, 'r') as f:
        rows = f.readlines()
    print(f'number of data: {len(rows)}')

    with open(save_file, 'w') as f:
        for row in rows:
            row = json.loads(row)
            conversation = row.pop('conversation')

            conversations = []
            for conv in conversation:
                human = conv['human'].strip()
                assistant = conv['assistant'].strip()
                conversations.append({'role': 'user', 'content': human})
                conversations.append({'role': 'assistant', 'content': assistant})
            row['conversations'] = conversations

            row = json.dumps(row, ensure_ascii=False)
            f.write(f'{row}\n')


if __name__ == '__main__':
    firefly2chatglm3()


