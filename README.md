# Firefly: 大模型训练工具

<div align="left">

![GitHub Repo stars](https://img.shields.io/github/stars/yangjianxin1/Firefly?style=social)
[![Generic badge](https://img.shields.io/badge/微信交流群-Firefly-brightgreen?logo=wechat)](./pics/wechat-group.jpeg)
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/YeungNLP)

[//]: # ([![Generic badge]&#40;https://img.shields.io/badge/微信-Firefly-brightgreen?logo=wechat&#41;]&#40;./pics/wechat.jpeg&#41;)
</div>

<img src="pics/firefly_logo.png" width="250">

欢迎加入Firefly大模型技术交流群，关注我们的公众号，点击加群按钮即可。

<img src="pics/gongzhonghao.png" width="300">

欢迎关注我们的知乎进行交流讨论：**[红雨瓢泼](https://www.zhihu.com/people/jian-xin-15-96)**

## 项目简介
**Firefly** 是一个开源的大模型训练项目，支持对主流的大模型进行预训练和指令微调，包括但不限于MiniCPM、Llama、InternLM、Baichuan、ChatGLM、Yi、Deepseek、Qwen、Orion、Ziya、Xverse、Mistral、Mixtral-8x7B、Zephyr、Vicuna、Bloom等。
本项目支持全量参数训练和QLoRA高效训练，支持预训练和指令微调。 如果你的训练资源有限，我们极力推荐使用QLoRA进行指令微调，因为我们在Open LLM Leaderboard上验证了该方法的有效性，并且取得了非常不错的成绩。

🔔 本项目主要内容如下：
- 📗 支持预训练和指令微调，支持全量参数训练和QLoRA高效训练。通过配置文件的方式训练不同的模型，小白亦可快速上手训练模型。
- 📗 支持绝大部分主流的开源大模型，如MiniCPM、Llama、InternLM、Baichuan、ChatGLM、Yi、Deepseek、Qwen、Orion、Ziya、Xverse、Mistral、Mixtral-8x7B、Zephyr、Vicuna、Bloom，训练时与各个官方的chat模型的template对齐。
- 📗 整理并开源指令微调数据集：firefly-train-1.1M 、moss-003-sft-data、ultrachat、 WizardLM_evol_instruct_V2_143k、school_math_0.25M。
- 📗 开源[Firefly系列指令微调模型权重](https://huggingface.co/YeungNLP) 。
- 📗 在Open LLM Leaderboard上验证了QLoRA训练流程的有效性。

当前版本针对不同的chat模型的template进行了适配，代码存在较大的更新。若你更喜欢此前的版本，可下载代码[v0.0.1-alpha](https://github.com/yangjianxin1/Firefly/releases/tag/v0.0.1-alpha)

## News
- 🔥 优化训练流程，支持全量训练和QLoRA高效训练，支持预训练和指令微调。指令微调的template与原有的chat模型对齐，支持绝大多数开源模型，包括MiniCPM、Llama、InternLM、Baichuan、ChatGLM、Yi、Deepseek、Qwen、Orion、Ziya、Xverse、Mistral、Mixtral-8x7B、Zephyr、Vicuna、Bloom等。
- 🔥 开源模型权重[firefly-mixtral-8x7b](https://huggingface.co/YeungNLP/firefly-mixtral-8x7b) ，在[🤗Open LLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)分数为70.34，超越Yi-34B、Llama2-65B-Chat、Qwen-14B、Vicuna-33B-v1.3等模型。
- 🔥 开源[LongQLoRA](https://github.com/yangjianxin1/LongQLoRA)， 【[技术报告](https://arxiv.org/abs/2311.04879)】。可高效扩展LLama上下文长度，在单张32GB V100上将Llama2长度扩展至8k（亦可扩展至12k），仅微调1000 step，在PG19和Proof-pile数据集上的perplexity优于LongLoRA，在PG19上略胜MPT-7B-8K。
- 🔥 开源[Firefly-LLaMA2-Chinese项目](https://github.com/yangjianxin1/Firefly-LLaMA2-Chinese)，**在4*V00上进行高效训练**，经过中文词表扩充、增量预训练、多轮指令微调，在CMMLU上超越Linly、Yayi、FlagAlpha等，与Ziya、Chinese-Alpaca表现基本持平。
- 🔥 开源[firefly-baichuan2-13b](https://huggingface.co/YeungNLP/firefly-baichuan2-13b)，在OpenCompass的CMMLU榜单上以56.83的分数，位列第8，比百川官方Chat模型略低1.57分。
- 🔥 开源[firefly-llama-30b](https://huggingface.co/YeungNLP/firefly-llama-30b)，在[🤗Open LLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)上以64.83分，同量级模型**排名第10**。
- 🔥 开源[firefly-llama2-13b](https://huggingface.co/YeungNLP/firefly-llama2-13b)，在[🤗Open LLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)上以62分，同量级模型**排名第3**，比榜首略低0.5分。
- 🔥 开源[firefly-llama-13b](https://huggingface.co/YeungNLP/firefly-llama-13b)，在[Hugging Face的Open LLM排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)上复刻Vicuna-13B，比Vicuna-13b-1.1略高0.2分，比llams-2-13b-chat略低0.5分。
- [LLMPruner：大语言模型裁剪工具](https://github.com/yangjianxin1/LLMPruner) ，开源[裁剪后的Bloom模型权重](https://huggingface.co/YeungNLP) 。

## 相关项目

| Project                                                                                        | Description                                                                                    |
|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [Firefly-LLaMA2-Chinese](https://github.com/yangjianxin1/Firefly-LLaMA2-Chinese)  | Firefly轻量级增量预训练项目，支持使用QLoRA增量预训练Baichuan2、Llama2、Llama、Falcon、Qwen、Baichuan、InternLM、Bloom等大模型 |
| [LongQLoRA](https://github.com/yangjianxin1/LongQLoRA) | 大模型长度扩展项目，可在单卡V100上将LLaMA-13B的长度扩展至8192，且性能逼近MPT-8K                                            |



## 技术博客
<details><summary><b>技术博客</b></summary>

- [图解大模型推理优化之KV Cache](https://mp.weixin.qq.com/s/7Fm8LbUN9jQ2HqxPbUU7UQ)
- [Mixtral-8x7B MoE大模型微调实践，超越Llama2-65B](https://mp.weixin.qq.com/s/f24e-Tp-1WyXTbVOzePvhg)
- [LongQLoRA：单卡高效扩展LLaMA2-13B的上下文长度](https://mp.weixin.qq.com/s/lptWXi9sZXd2MTTXZsDiPw)
- [详解基于调整RoPE旋转角度的大模型长度外推方法](https://mp.weixin.qq.com/s/RtI95hu-ZLxGkdGuNIkERQ)
- [图解RoPE旋转位置编码及其特性](https://mp.weixin.qq.com/s/-1xVXjoM0imXMC7DKqo-Gw)
- [QLoRA轻量级增量预训练方案，及汉化Llama2的实践](https://mp.weixin.qq.com/s/26-Qxma9M2wGoTQgOlKRmQ)
- [Firefly多轮对话微调书生·浦语InternLM-7B实践](https://mp.weixin.qq.com/s/98OLdkHjoGDHNDbYL7RerA)
- [🤗Firefly微调LLaMA-30B，Open LLM榜单同量级第10名](https://mp.weixin.qq.com/s/fFT0Pxfecma4n_fXQYb2Mw)
- [通义千问Qwen-7B效果如何？Firefly微调实践，效果出色](https://mp.weixin.qq.com/s/5OAx83j6Op299XAfa496ww)
- [源码解析ChatGLM2多轮对话训练方法的不足，以及改进方法](https://mp.weixin.qq.com/s/nhogoWnzl3nrs_77r38_UA)
- [Firefly增强Baichuan-13B的多轮对话能力](https://mp.weixin.qq.com/s/djO8Tg3emmy6wzw_rTUlcw)
- [🤗Open LLM排行榜，firefly-llama2-13b在所有13B模型中排名第三，比榜首略低0.5分](https://mp.weixin.qq.com/s/w1V3QGvsRTQsQqAKp2z6Kg)
- [百万数据增强Baichuan-13B的多轮对话能力](https://mp.weixin.qq.com/s/djO8Tg3emmy6wzw_rTUlcw)
- [Firefly单卡复刻Vicuna-13B，Open LLM榜单🤗略高0.2分](https://mp.weixin.qq.com/s/QG2YMo_QxaxS_Rr2yJrIeA)
- [微调百川Baichuan-13B保姆式教程，手把手教你训练百亿大模型](https://mp.weixin.qq.com/s/ZBY6kbogHjbCQvZBzNEqag)
- [Firefly-Ziya-13B开源，QLoRA+百万数据，单卡可训百亿大模型](https://mp.weixin.qq.com/s/vgNK6D-_0j4Chk2H1Ev-Ig)
- [Firefly｜百川baichuan-7B实测，QLoRA+百万指令数据微调](https://mp.weixin.qq.com/s/_eTkDGG5DmxyWeiQ6DIxBw)
- [Firefly | QLoRA+百万数据，多卡高效微调bloom-7b1模型](https://mp.weixin.qq.com/s/lA4YUJ9XGpKlUUUjz0Le-g)
- [QLoRA文章解读 & 单卡高效微调bloom-7b1](https://mp.weixin.qq.com/s/DED7yeiE0DibsVzTmMeDOw)
- [Firefly(流萤): 中文对话式大语言模型](https://mp.weixin.qq.com/s/TX7wj8IzD_EaMTvk0bjRtA)
- [LLMPruner：大语言模型裁剪工具](https://mp.weixin.qq.com/s/leVtrwZc1zLput51Nr99lw)

</details>


## 模型评测

### Open LLM Leaderboard评测
评测结果来源于Hugging Face的[Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)。我们的模型均采用QLoRA脚本进行训练，训练仅使用1~2张V100。


| 模型                          | Average | ARC   | HellaSwag | MMLU  | TruthfulQA |
|-----------------------------|---------|-------|-----------|-------|------------|
| **firefly-mixtral-8x7b**    | 70.16   | 68.09 | 85.76     | 71.49 | 55.31      |
| Yi-34B-Chat                 | 69.97   | 65.44 | 84.16     | 74.9  | 55.37      |
| **firefly-llama-30b**       | 64.83   | 64.25 | 83.64     | 58.23 | 53.2       |
| falcon-40b-instruct         | 63.47   | 61.6  | 84.31     | 55.45 | 52.52      |
| guanaco-33b                 | 62.98   | 62.46 | 84.48     | 53.78 | 51.22      |
| **firefly-llama2-13b-v1.2** | 62.17   | 60.67 | 80.46     | 56.51 | 51.03      |
| **firefly-llama2-13b**      | 62.04   | 59.13 | 81.99     | 55.49 | 51.57      |
| vicuna-13b-v1.5             | 61.63   | 56.57 | 81.24     | 56.67 | 51.51      |
| mpt-30b-chat                | 61.21   | 58.7  | 82.54     | 51.16 | 52.42      |
| wizardlm-13b-v1.2           | 60.79   | 59.04 | 82.21     | 54.64 | 47.27      |
| vicuna-13b-v1.3             | 60.01   | 54.61 | 80.41     | 52.88 | 52.14      |
| llama-2-13b-chat            | 59.93   | 59.04 | 81.94     | 54.64 | 44.12      |
| vicuna-13b-v1.1             | 59.21   | 52.73 | 80.14     | 51.9  | 52.08      |
| guanaco-13b                 | 59.18   | 57.85 | 83.84     | 48.28 | 46.73      |


## 模型列表

🔔 使用本项目的训练代码，以及上述训练数据，我们训练并开源了以下模型权重。

中文模型：

| 模型                                                                             | 基座模型                                | 训练长度 |
|--------------------------------------------------------------------------------|-------------------------------------|------|
| [firefly-baichuan2-13b](https://huggingface.co/YeungNLP/firefly-baichuan2-13b) | baichuan-inc/Baichuan2-13B-Base     | 1024 |  
| [firefly-baichuan-13b](https://huggingface.co/YeungNLP/firefly-baichuan-13b)   | baichuan-inc/Baichuan-13B-Base      | 1024 |  
| [firefly-qwen-7b](https://huggingface.co/YeungNLP/firefly-qwen-7b)             | Qwen/Qwen-7B                        | 1024 |  
| [firefly-chatglm2-6b](https://huggingface.co/YeungNLP/firefly-chatglm2-6b)     | THUDM/chatglm2-6b                   | 1024 |  
| [firefly-internlm-7b](https://huggingface.co/YeungNLP/firefly-internlm-7b)     | internlm/internlm-7b                | 1024 |  
| [firefly-baichuan-7b](https://huggingface.co/YeungNLP/firefly-baichuan-7b)     | baichuan-inc/baichuan-7B            | 1024 |           
| [firefly-ziya-13b](https://huggingface.co/YeungNLP/firefly-ziya-13b)           | YeungNLP/Ziya-LLaMA-13B-Pretrain-v1 | 1024 |           
| [firefly-bloom-7b1](https://huggingface.co/YeungNLP/firefly-bloom-7b1)         | bigscience/bloom-7b1                | 1024 |
| [firefly-bloom-2b6-v2](https://huggingface.co/YeungNLP/firefly-bloom-2b6-v2)   | YeungNLP/bloom-2b6-zh               | 512  |
| [firefly-bloom-2b6](https://huggingface.co/YeungNLP/firefly-bloom-2b6)         | YeungNLP/bloom-2b6-zh               | 512  |
| [firefly-bloom-1b4](https://huggingface.co/YeungNLP/firefly-bloom-1b4)         | YeungNLP/bloom-1b4-zh               | 512  |


英文模型：

| 模型                                                                     | 基座模型              | 训练长度 |
|------------------------------------------------------------------------|-------------------|------|
| [firefly-mixtral-8x7b](https://huggingface.co/YeungNLP/firefly-mixtral-8x7b)    | mistralai/Mixtral-8x7B-v0.1                  | 1024 |
| [firefly-llama-30b](https://huggingface.co/YeungNLP/firefly-llama-30b) | huggyllama/llama-30b | 1024 |  
| [firefly-llama-13-v1.2](https://huggingface.co/YeungNLP/firefly-llama2-13b-v1.2) | NousResearch/Llama-2-13b-hf | 1024 |  
| [firefly-llama2-13b](https://huggingface.co/YeungNLP/firefly-llama2-13b) | NousResearch/Llama-2-13b-hf | 1024 |           
| [firefly-llama-13b-v1.2](https://huggingface.co/YeungNLP/firefly-llama-13b-v1.2) | huggyllama/llama-13b | 1024 |           
| [firefly-llama-13b](https://huggingface.co/YeungNLP/firefly-llama-13b) | huggyllama/llama-13b | 1024 |



## 训练数据
### 指令微调数据
🔔 目前本项目主要整理了如下指令数据集，并将其整理成统一的数据格式：

| 数据集                                                                                                           | 介绍                                                                                                      |
|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| [firefly-train-1.1M](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M)                             | 我们收集了23种常见的中文NLP任务的数据，并且构造了许多与中华文化相关的数据，如对联、作诗、文言文翻译、散文、金庸小说等。对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万 |
| [moss-003-sft-data](https://huggingface.co/datasets/YeungNLP/moss-003-sft-data)                               | 由复旦大学MOSS团队开源的中英文多轮对话数据，包含100万+数据                                                                       |
| [ultrachat](https://huggingface.co/datasets/YeungNLP/ultrachat)                                               | 由清华大学开源的英文多轮对话数据，包含140万+数据                                                                              |
| [WizardLM_evol_instruct_V2_143k](https://huggingface.co/datasets/YeungNLP/WizardLM_evol_instruct_V2_143k) | 由WizardLM项目开源的英文指令微调数据集，通过Evol-Instruct方法让指令进化，加强指令的复杂度，以提升模型对复杂指令的遵循能力。包含143k条数据。                      |
| [school_math_0.25M](https://huggingface.co/datasets/YeungNLP/school_math_0.25M)      | 由BELLE项目组开源的数学运算指令数据，包含25万条数据。                                                                          |
| [shareAI/CodeChat](https://huggingface.co/datasets/shareAI/CodeChat)      | 主要包含逻辑推理、代码问答、代码生成相关语料样本。                                                                          |
| [shareAI/ShareGPT-Chinese-English-90k](https://huggingface.co/datasets/shareAI/ShareGPT-Chinese-English-90k)      | 中英文平行双语优质人机问答数据集，覆盖真实复杂场景下的用户提问。                                                                          |

可按需选择、采样或组合上述数据。若是训练中文模型，建议尝试使用moss数据，英文可尝试ultrachat和WizardLM数据。

训练数据为jsonl格式，每一行的数据格式如下，其中conversation字段是必需的，可以根据实际需求添加或删除其他字段。
可以参考项目中的data/dummy_data.jsonl文件。
```json
{
    "conversation_id":44,
    "category":"Brainstorming",
    "conversation":[
        {
            "human":"怎么能够有效地提升个人能力？",
            "assistant":"提升个人能力可以通过以下几个方面实现：\n\n1. 学习新知识和技能：阅读相关书籍、参加课程和工作坊、观看教育视频等学习方式，可以帮助你掌握新的知识和技能，从而提升个人能力。\n\n2. 持续练习和实践：不断地练习和实践所学的知识和技能，可以帮助你更好地掌握和应用这些技能，提高你的表现和效率。\n\n3. 寻求反馈和改进：寻求他人的反馈和建议可以帮助你发现自己的不足之处，进一步完善和改进自己的能力。\n\n4. 建立良好的工作和生活习惯：建立良好的工作和生活习惯，如合理安排时间、注意健康、保持积极态度等，可以提高你的工作效率和生活品质，从而提升个人能力。\n\n5. 社交和交流：与他人交流和合作，可以帮助你更好地了解他人的观点和经验，从而提升自己的能力和见识。"
        },
        {
            "human":"对于初学者来说，如何快速掌握编程技能？",
            "assistant":"以下是几个建议，可以帮助初学者快速掌握编程技能：\n\n1. 学习基础知识：建议先学习编程语言的基础知识和语法，例如变量、数据类型、控制结构、函数等。可以通过在线课程、编程书籍、视频教程等方式学习。\n\n2. 实践编程：进行实际的编程练习是非常重要的，可以通过参加编程挑战、编写小项目等方式来进行实践。\n\n3. 寻求反馈：可以寻求他人的反馈和建议，例如在线编程社区、编程导师等，他们可以帮助你发现问题并及时纠正。\n\n4. 阅读优秀代码：阅读和分析优秀的代码可以帮助你学习其他程序员的编程思路和技巧，从而提高自己的编程能力。\n\n5. 不断学习和更新：编程技术和工具在不断更新和发展，因此需要不断学习新的知识和技能，例如学习新的编程语言、框架、工具等，以跟上技术的发展和变化。\n\n总之，学习编程需要耐心和坚持，并且需要不断地实践和学习。通过以上建议，可以帮助初学者快速掌握编程技能。"
        },
        {
            "human":"除了参加编程挑战和阅读代码，还有哪些方式可以提升自己的编程能力呢？",
            "assistant":"除了参加编程挑战和阅读代码，还有以下几种方式可以提升自己的编程能力：\n\n1. 参与开源项目：加入一个开源项目，可以让你了解到实际的软件开发流程和团队合作方式，并且可以与其他开发人员合作，学习和分享编程技巧和经验。\n\n2. 进行代码重构：对已经编写的代码进行重构，可以帮助你了解更好的编程实践和设计模式，从而提高代码的可读性和可维护性。\n\n3. 练习算法和数据结构：算法和数据结构是编程中的重要基础，练习算法和数据结构可以帮助你了解不同的数据结构和算法，从而更好地解决实际问题。\n\n4. 学习软件工程和架构：学习软件工程和架构的知识，可以帮助你了解如何设计和构建可靠、可扩展和可维护的软件系统。\n\n5. 参加编程社区和论坛：参加编程社区和论坛可以让你与其他开发人员分享和讨论编程经验和技巧，了解最新的技术趋势和解决方案。\n\n总之，提升编程能力需要不断地学习和实践，并与其他开发人员和编程专家进行交流和合作，以获取更多的经验和知识。"
        },
        {
            "human":"你有没有推荐的编程挑战网站？",
            "assistant":"当然，以下是一些值得推荐的编程挑战网站：\n\n1. HackerRank：HackerRank 是一个全球领先的技术招聘平台，提供数百种编程挑战和比赛，包括算法、数据结构、数学、SQL、AI 等多个技术领域。\n\n2. LeetCode：LeetCode 是一个在线编程学习和实践平台，提供数千道编程题目，主要涵盖算法和数据结构。\n\n3. Codeforces：Codeforces 是一个全球知名的编程竞赛网站，提供数百个比赛和练习题目，主要涵盖算法和数据结构。\n\n4. Codewars：Codewars 是一个在线编程学习和实践平台，提供数千道编程题目，主要涵盖算法、数据结构、面向对象编程等多个技术领域。\n\n5. Project Euler：Project Euler 是一个非营利性的在线编程挑战平台，提供一系列数学和计算机科学问题，需要用编程语言进行解答。\n\n以上这些编程挑战网站可以帮助你提高编程能力，并且可以与其他编程爱好者进行交流和竞争。"
        },
        {
            "human":"在学习编程的过程中，如何避免陷入困境和放弃呢？",
            "assistant":"学习编程是一个需要长期坚持和不断学习的过程，以下是一些避免陷入困境和放弃的方法：\n\n1. 制定合理的学习计划：制定合理的学习计划，包括学习时间、学习内容、目标等，可以帮助你更好地管理时间和精力，避免学习过程中的松懈和分心。\n\n2. 寻找合适的学习资源：选择适合自己的学习资源，例如在线课程、编程书籍、视频教程等，可以让你更好地了解和掌握编程知识和技能。\n\n3. 寻求帮助和支持：在学习过程中，遇到问题和困难是很正常的，可以寻求他人的帮助和支持，例如参加编程社区、找到编程导师等。\n\n4. 进行实践和项目：实践和项目是学习编程的重要组成部分，可以帮助你更好地了解和掌握编程技能，同时也可以提高学习的兴趣和动力。\n\n5. 坚持并保持兴趣：坚持学习和保持兴趣是学习编程的关键。可以通过参加编程社区、参加编程竞赛、与其他编程爱好者交流等方式来保持兴趣和动力。\n\n总之，学习编程需要耐心和坚持，并需要不断学习和实践。通过以上方法可以帮助你避免陷入困境和放弃。"
        }
    ],
}
```

其中firefly-train-1.1M的数据分布如下图所示：

<img src="pics/task_distribution.png" width="380"> 

### 预训练数据
数据格式可以参考项目中的data/pretrain/dummy_pretrain.jsonl文件。

## 模型训练
若训练中报错，可先查看[FAQ]()。

我们将训练中使用的各种组件抽取出来，以便后续的扩展和优化，详见component目录下的实现。训练时的参数配置存储在train_args目录下，方便统一管理和更改。大家可以在train_args目录下查看不同模型的训练配置，按需修改或添加。

### 安装环境
在requirements.txt下固定了几个主要的python包的版本，执行如下脚本即可。注意：
- 使用QLoRA训练Baichuan2时，需要安装torch==2.0，并且卸载xformers。对于其他模型，我们均在torch==1.13上进行训练。
- 使用QLoRA训练Qwen时，需将flash-attn卸载，否则会报错。
```bash
pip install requirements.txt
```

### 损失函数
预训练时，我们采用经典的自回归损失，即每个位置的token都会参与loss计算。

指令微调时，我们仅计算assistant回复部分的loss。

### 参数说明
📝 train_args目录下存储了不同模型使用不同训练方式的配置文件，主要参数说明如下：
- output_dir：训练输出目录，存储checkpoint、tokenizer、tensorboard等
- model_name_or_path：预训练模型的本地目录，或者在huggingface上的模型名称。
- train_file：训练数据集路径。sft时，需要设置为文件，可以使用data/dummy_data.jsonl进行debug。pretrain时，需要设置为目录。脚本会自动扫描目录下的所有jsonl文件。
- template_name：指令微调时，使用的模板名称。
- num_train_epochs：训练的轮次。如果数据量足够大，一般建议只训一个epoch。
- tokenize_num_workers：预训练时，tokenize的线程数，默认为10。
- deepspeed：deepspeed的训练配置文件。全量参数训练时，将采用deepspeed，关于deepspeed的参数配置说明，请参考[deepspeed文档](https://hf-mirror.com/docs/transformers/main/en/deepspeed#deepspeed)
- train_mode：训练模式，qlora或者full，默认为qlora。
- task_type：任务类型，pretrain或者sft，默认为sft。
- per_device_train_batch_size：每张显卡的batch size。
- gradient_accumulation_steps：梯度累计步数。global batch=num_gpus * per_device_train_batch_size * gradient_accumulation_steps。
- gradient_checkpointing：如果显存捉襟见肘，可以开启。以时间换空间，模型不缓存激活状态，会进行两次forward计算，以节省显存。
- learning_rate：学习率。全量参数微调的时候，建议小一些，1e-5或5e-6。
- max_seq_length：训练时的最大长度。按照自己的设备进行设置，越长需要占用越多显存。
- logging_steps：每隔多少步统计一次train loss。
- save_steps：每隔多少步保存一个模型。
- save_total_limit：output_dir目录中最多保存多少个checkpoint，超出则会将最旧的删除。
- lr_scheduler_type：学习率变化策略。
- warmup_steps：warm up步数。学习率经过多少步，增长到指定的数值。
- optim：优化器。如果是全量参数微调，建议使用adamw_hf。
- seed：随机种子，用于复现实验结果。
- fp16：使用使用fp16混合精度。V100建议开启。
- bf16：使用使用bf16混合精度。A100建议开启。

以下几个参数，当使用QLoRA训练的时候，需要设置：
- lora_rank：qlora矩阵的秩。一般设置为8、16、32、64等，在qlora论文中作者设为64。越大则参与训练的参数量越大，一般来说效果会更好，但需要更多显存，。
- lora_alpha: qlora中的缩放参数。一般设为16、32即可。
- lora_dropout: lora权重的dropout rate。

关于deepspeed的参数配置，可按需自行修改。


### 开始训练

💻 全量参数预训练，将{num_gpus}替换为显卡数量：
```bash
deepspeed --num_gpus={num_gpus} train.py --train_args_file train_args/pretrain/full/bloom-1b1-pretrain-full.json
```

💻 全量参数指令微调，将{num_gpus}替换为显卡数量：
```bash
deepspeed --num_gpus={num_gpus} train.py --train_args_file train_args/sft/full/bloom-1b1-sft-full.json
```

💻 单卡QLoRA预训练：
```bash
python train.py --train_args_file train_args/pretrain/qlora/yi-6b-pretrain-qlora.json
```

💻 单卡QLoRA指令微调：
```bash
python train.py --train_args_file train_args/sft/qlora/yi-6b-sft-qlora.json
```

💻 多卡QLoRA预训练：
```bash
torchrun --nproc_per_node={num_gpus} train.py --train_args_file train_args/pretrain/qlora/yi-6b-pretrain-qlora.json
```

💻 多卡QLoRA指令微调：
```bash
torchrun --nproc_per_node={num_gpus} train.py --train_args_file train_args/sft/qlora/yi-6b-sft-qlora.json
```

## 模型使用

### 权重合并
如果使用LoRA或者QLoRA进行训练，本项目仅保存adapter的权重和配置文件，需要将adapter权重与base model进行合并。脚本见script/merge_lora.py

### 模型推理
我们提供了多轮对话的交互脚本，详见script/chat目录，该脚本可同时兼容本项目训练的所有模型进行推理。脚本中设置的template_name，需要与模型训练时的template_name一致。
```bash
cd script/chat
python chat.py
```

生成脚本中的top_p、temperature、repetition_penalty、do_sample等参数对模型的生成效果影响较大，可按照自己的使用场景进行调试修改。

推理脚本中支持使用base model和adapter进行推理，缺点是每次启动脚本都需要合并一次权重，等待时间较久。

支持使用4bit进行推理，显存要求低，效果会略有下降。


## FAQ
#### 问题1：OOM如何解决？
如果发生OOM，可以缩小per_device_train_batch_size、max_seq_length等参数来缓解。也可以设gradient_checkpointing=true，可以大幅降低显存占用，但训练速度会变慢一些。

#### 问题2：安装包错误
requirements.txt中有各python包的版本
```bash
pip install -r requirements.txt
```

#### 问题3：是否支持DeepSpeed+QLoRA？
我们尝试过DeepSpeed+QLoRA的训练策略，但尚未成功，目前建议使用torchrun启动训练。后续若成功，我们将更新代码。

#### 问题4：如何指定使用某些卡训练？
通过如下方式，即可指定使用0和1号卡进行训练:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node={num_gpus} train_qlora.py --train_args_file train_args/qlora/baichuan-7b-sft-qlora.json
```

#### 问题5：训练Baichuan2失败
训练Baichuan2需要安装torch==2.0，并且卸载xformers，否则会报错
```
RuntimeError: No such operator xformers::efficient_attention_forward_generic - did you forget to build xformers with `python setup.py develop`?
```

#### 问题6：训练Qwen失败
Qwen进行QLoRA训练需要卸载flash-attn，否则会报错：
```
assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v))) 
```

#### 问题7：Qwen-Base和Yi-Base经过SFT之后，没法生成<|im_end|>，无法正常停止
经查询，该问题广泛存在于Qwen官方代码库的issue中，如果训练Qwen-Base和Yi-Base，建议设template_name="default"，可以避免该问题。
如果对Qwen-Chat和Yi-Chat模型进行SFT，则不会产生该问题，可将template_name分别设为"qwen"和"yi"。


## 局限性和使用限制
由于模型参数量限制、训练数据的清洗程度等因素，本项目开源的模型可能存在以下局限性：
- 对于事实性知识，容易产生错误的回复。
- 由于未经过无害化微调，可能会产生歧视、危害、违背伦理道德的言论。
- 在代码、推理上的能力仍有欠缺。

基于以上模型的局限性，我们要求本项目的代码、数据、模型不得用于对社会造成危害的用途，且应当遵循基座模型的商业许可。

## 引用
若使用本项目的数据、代码或模型，请引用本项目。
```text
@misc{Firefly,
  author = {Jianxin Yang},
  title = {Firefly(流萤): 中文对话式大语言模型},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yangjianxin1/Firefly}},
}
```

[//]: # (## 关注我们)

[//]: # ()
[//]: # (<img src="pics/gongzhonghao.jpeg" width="250"> )

## Star History
![Star History Chart](https://api.star-history.com/svg?repos=yangjianxin1/Firefly&type=Date)




