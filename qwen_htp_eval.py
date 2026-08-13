"""Zero-shot / few-shot evaluation of a Qwen-VL model on the HTP dataset.

Assumes a vLLM OpenAI-compatible server is already running, e.g.

    vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
        --served-model-name qwen-vl \
        --max-model-len 8192 \
        --limit-mm-per-prompt image=1 \
        --gpu-memory-utilization 0.9

Usage:
    python qwen_htp_eval.py --data-dir dataset/HTP --out output/qwen_htp.csv
    python qwen_htp_eval.py --mode direct --limit 20      # 快速冒烟测试
    python qwen_htp_eval.py --from-csv output/qwen_htp.csv  # 只重算指标

    # 抽客观视觉特征，再用 htp_probe.py 训逻辑回归
    python qwen_htp_eval.py --mode extract --out output/qwen_htp_feat.csv
    python htp_probe.py output/qwen_htp_feat.csv
"""

import argparse
import base64
import collections
import csv
import io
import itertools
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from openai import OpenAI

# HTP 的目录名是 00 / 01，这里必须写清楚每个标签的真实含义，
# 否则模型只是在猜两个没有语义的符号。
CLASS_DESC = {
    "00": "存在心理异常倾向",
    "01": "心理状态正常",
}
CLASS_DIRS = ["00", "01"]
CHOICES = [CLASS_DESC[c] for c in CLASS_DIRS]

# 最终答案用单个字母，这样引导解码后第一个 token 就是答案，
# 可以直接从 top_logprobs 里读出两个选项的概率，拿来算 AUC。
LETTERS = ["A", "B"]
LETTER2DIR = dict(zip(LETTERS, CLASS_DIRS))

ENV_NAME = {"00": "child", "01": "college", "02": "social"}

RULES = (
    "这是一张房树人（HTP，House-Tree-Person）绘画测验的作品，由被试手绘完成。\n"
    "你的任务是根据画面中实际可观察到的房屋、树木、人物以及整体构图特征，"
    "综合判断该作品更符合哪一种心理状态标签。\n"
    "\n"
    "请注意：判断时必须同时寻找支持“存在心理异常倾向”和支持“心理状态正常”的视觉证据，"
    "不能只寻找异常特征。"
    "“心理状态正常”不是在找不到异常时使用的默认标签，而是同样需要根据画面中的完整性、"
    "协调性、合理比例、稳定结构、自然表达和场景连贯性等积极特征进行判断。"
    "同样，“存在心理异常倾向”也不能因为出现一个轻微异常特征就直接选择。\n"
    "\n"
    "请在内部按照以下顺序分析："
    "整体画面 -> 房屋 -> 树木 -> 人物 -> 房树人之间的关系 -> 正常证据与异常证据对比 -> 最终判断。\n"
    "\n"
    "只分析图片中真正能够观察到的特征，不要猜测不存在或看不清的内容。"
    "绘画能力、年龄、简笔画风格、卡通风格、透视误差、图片裁剪、遮挡、扫描过淡、"
    "低分辨率等因素都可能影响画面表现，不能直接解释为心理异常。\n"
    "\n"
    "====================\n"
    "一、整体画面\n"
    "====================\n"
    "\n"
    "【支持心理状态正常的特征】\n"
    "1. 房屋、树木、人物三个核心对象均能够清楚识别，并且基本完整。\n"
    "2. 三个对象在画面中的大小适中，没有全部挤在极小区域，也没有明显超出或压迫整个画面。\n"
    "3. 房、树、人之间保持自然、合理的空间距离，没有明显孤立或被大片空白强行分隔。\n"
    "4. 三个对象能够形成基本连贯的场景，例如共享地面、方向一致、空间位置合理，"
    "整体看起来属于同一个环境，而不是三个互不相关的符号。\n"
    "5. 整体构图稳定，没有明显失衡、极端偏向某个角落或严重碎片化。\n"
    "6. 线条总体连续、自然、力度相对稳定，没有大量异常断裂、极端微弱或杂乱涂画。\n"
    "7. 画面具有适量细节。可以简单，但主要对象的基本结构能够表达清楚。\n"
    "8. 如果存在草地、道路、太阳、云、花、地面等环境元素，并且与主体自然结合，"
    "可以作为场景完整和组织良好的积极证据。\n"
    "9. 没有大面积、无明显绘画目的的异常涂黑、重阴影或局部反复强调。\n"
    "\n"
    "【支持存在心理异常倾向的特征】\n"
    "1. 房屋、树木、人物中的一个或多个核心对象完全缺失，且无法由裁剪、遮挡或画法解释。\n"
    "2. 整幅画极度简化，只剩极少线条，大量重要结构同时消失，整体信息量异常低。\n"
    "3. 房、树、人整体只占巨大画布中极小区域，周围存在大量空白。\n"
    "4. 三个对象之间距离极端遥远，某个对象明显孤立，或者整个场景严重碎片化。\n"
    "5. 对象像漂浮在不同区域，彼此几乎没有空间联系，整体场景严重不连贯。\n"
    "6. 存在大量非常微弱、间断、犹豫、僵硬或严重潦草的线条。\n"
    "7. 存在异常大面积涂黑、重度阴影或某个对象被明显过度强调。\n"
    "\n"
    "普通简单构图、没有背景装饰、使用简笔画方式，本身都不能直接作为异常证据。\n"
    "\n"
    "====================\n"
    "二、房屋\n"
    "====================\n"
    "\n"
    "【支持心理状态正常的特征】\n"
    "1. 房屋能够清楚识别，墙体、屋顶等主要结构完整。\n"
    "2. 房屋通常具有合理的入口结构，例如门能够正常辨认；如果当前视角合理地看不到门，"
    "也不能因此判断异常。\n"
    "3. 窗户存在且位置、数量和尺寸基本合理；如果画法本身非常简洁，没有画窗户也不能单独判断异常。\n"
    "4. 房屋整体大小适中，与人物、树木以及画布比例基本协调。\n"
    "5. 墙体和屋顶结构稳定，没有明显倒塌、断裂或严重几何变形。\n"
    "6. 普通正面房屋、二维画法、简单几何房屋均可以是正常表现，"
    "不要求绘画具备专业透视能力。\n"
    "7. 屋顶、烟囱、门窗等细节如果自然、适量并与房屋结构协调，可以作为完整性证据。\n"
    "8. 即使存在烟囱、屋顶装饰或少量阴影，只要程度自然且具有明确绘画意义，也不应视为异常。\n"
    "\n"
    "【支持存在心理异常倾向的特征】\n"
    "1. 房屋结构较完整、可见方向正常，但明显选择性缺少门。\n"
    "2. 房屋其他部分较详细，但明显选择性缺少所有窗户。\n"
    "3. 房屋相对于画布以及其他对象极端微小或被严重边缘化。\n"
    "4. 墙体、屋顶整体严重倾斜，表现出明显不稳定或坍塌感，且不能由透视解释。\n"
    "5. 屋顶与墙体断开、几何结构极端扭曲、建筑组织明显不可能或严重怪异。\n"
    "6. 房屋被过度简化到主要结构难以辨认。\n"
    "7. 屋顶存在异常密集、反复、明显失衡的装饰或强调。\n"
    "8. 烟囱产生大量、明显、被特别强调的烟雾。\n"
    "9. 墙壁存在没有明显绘画原因的大面积重度涂黑或阴影。\n"
    "\n"
    "普通房屋造型、简单屋顶、存在烟囱、普通二维画法、少量阴影等都不能单独作为异常依据。\n"
    "\n"
    "====================\n"
    "三、树木\n"
    "====================\n"
    "\n"
    "【支持心理状态正常的特征】\n"
    "1. 树干、树冠、树枝等主要部分能够正常识别，整体结构基本完整。\n"
    "2. 树木大小与房屋、人物和画布基本协调，不存在极端小型化。\n"
    "3. 树干和树冠之间连接自然，比例基本合理。\n"
    "4. 树枝向外自然展开，形态可以简单或风格化，但总体仍具有正常树木结构。\n"
    "5. 树冠可以是圆形、椭圆形、不规则形等，只要没有明显严重压缩、断裂或结构畸形，都可视为正常变化。\n"
    "6. 是否画叶子不能单独决定正常或异常，季节性、绘画风格和简化画法都可能导致无叶树。\n"
    "7. 普通树根、树皮、树疤、深色树干或自然阴影都可以属于正常绘画表现。\n"
    "\n"
    "【支持存在心理异常倾向的特征】\n"
    "1. 树木相对于其他主体和画布极端微小。\n"
    "2. 树干或树冠出现明显、突然、没有合理原因的截断。\n"
    "3. 树木明确表现为死亡、严重枯萎、破损或大量枝干折断，而不是单纯没有叶子。\n"
    "4. 树冠被异常压缩或严重扁平。\n"
    "5. 树干、树冠、树枝之间存在明显不可能的连接关系、严重扭曲或结构断裂。\n"
    "6. 出现大量非常尖锐、尖刺状并被明显强调的树枝。\n"
    "7. 根系被极端扩大、反复描画或成为异常突出的视觉中心。\n"
    "\n"
    "普通树根、少量尖枝、没有树叶、简单树冠、深色树干等不能单独判断为异常。\n"
    "\n"
    "====================\n"
    "四、人物\n"
    "====================\n"
    "\n"
    "【支持心理状态正常的特征】\n"
    "1. 人物头部、躯干、手臂和腿等主要身体结构基本完整。\n"
    "2. 人物各身体部分连接关系自然，没有明显断裂、错位或无法解释的结构缺失。\n"
    "3. 人物整体比例基本合理。允许存在普通儿童画、简笔画、卡通画中的比例误差，"
    "不要求符合严格人体解剖比例。\n"
    "4. 人物大小与房屋、树木和整个画布基本协调，不存在极端小型化。\n"
    "5. 如果人物尺寸允许，眼睛、鼻子、嘴等主要面部特征基本存在；"
    "如果人物很小，没有绘制完整五官也可以是正常表现。\n"
    "6. 人物采用火柴人、单线手臂或单线腿的简单风格时，如果整个人物风格统一并且结构完整，"
    "不能因为简单就判断异常。\n"
    "7. 人物可以正面、侧面或其他姿势，只要身体结构基本合理，都可以属于正常表现。\n"
    "8. 普通衣服填色、头发填色或身体局部自然阴影不能作为异常证据。\n"
    "9. 面部表情自然、中性、微笑或其他与场景协调的普通表情，可以支持正常判断。\n"
    "10. 即使人物没有明显表情，只要画法简单且没有其他异常支持，也不能因为表情中性就判断异常。\n"
    "\n"
    "【支持存在心理异常倾向的特征】\n"
    "1. 人物明显没有完成，例如主要身体区域完全缺失、头部缺失、躯干缺失、"
    "多个必要身体部分缺失，或者身体不同部分明显断开。\n"
    "2. 人物其他部分较详细且尺寸足够大，但眼睛、鼻子、嘴等主要面部结构被选择性大量省略。\n"
    "3. 一个或多个主要手臂或腿明显缺失，并且不能由姿势、透视或遮挡解释。\n"
    "4. 人物其他部分较详细，但四肢被异常简化为孤立单线。\n"
    "5. 头部、躯干、手臂、腿、手等之间存在非常明显且严重的比例失调。\n"
    "6. 人体结构出现不可能的连接方式、严重错位、明显断裂或大幅扭曲。\n"
    "7. 人物相对于房屋、树木和画布极端微小或明显被孤立。\n"
    "8. 整个人物或某个身体区域被异常大量、重度、反复涂黑，"
    "且不能用衣物或正常填色解释。\n"
    "9. 能够清楚识别出明显的悲伤、愤怒、痛苦、恐惧等强烈负面表情。\n"
    "10. 能够明确辨认人物画出了被突出表现的紧握拳头。\n"
    "\n"
    "人物不够漂亮、比例不精确、没有手指、画成火柴人、采用侧面姿势、衣服简单、"
    "发型特殊等情况，本身都不能作为心理异常依据。\n"
    "\n"
    "====================\n"
    "五、房屋、树木和人物之间的关系\n"
    "====================\n"
    "\n"
    "【支持心理状态正常的特征】\n"
    "1. 房屋、树木和人物的相对大小基本合理，没有非常极端的尺寸差异。\n"
    "2. 三者之间保持自然距离，没有某个对象被极端隔离。\n"
    "3. 对象具有基本共同空间，例如共享地面、位置关系能够形成一个完整场景。\n"
    "4. 人物能够自然地处于房屋和树木形成的环境中，而不是完全脱离其他对象。\n"
    "5. 即使对象之间没有互动，只要整体空间组织清楚、稳定、协调，也属于正常构图。\n"
    "\n"
    "【支持存在心理异常倾向的特征】\n"
    "1. 人物极端微小并且距离房屋、树木非常远。\n"
    "2. 房屋、树木和人物分别孤立在互不相关的区域。\n"
    "3. 三个对象的尺寸比例严重不协调。\n"
    "4. 不同对象之间在空间上和语义上明显断开，无法组成基本连贯场景。\n"
    "5. 多种空间关系异常同时存在，例如尺寸异常、隔离、大片空白和场景破碎共同出现。\n"
    "\n"
    "====================\n"
    "六、如何综合正常证据和异常证据\n"
    "====================\n"
    "\n"
    "请分别在内部建立两组证据："
    "一组是支持“心理状态正常”的证据，另一组是支持“存在心理异常倾向”的证据。"
    "最终比较两组证据的强度、可靠性、独立性和一致性，而不是只统计异常项目的数量。\n"
    "\n"
    "正常证据包括但不限于："
    "核心对象完整、结构稳定、比例大致协调、大小适中、线条自然连续、"
    "对象关系合理、整体场景连贯、细节适量、人物身体完整、房屋和树木结构自然、"
    "不存在明显选择性结构缺失等。\n"
    "\n"
    "异常证据包括但不限于："
    "核心对象缺失、严重结构缺失、明显选择性缺失、极端比例异常、"
    "严重结构扭曲、极端小型化、异常隔离、大面积异常涂黑、"
    "严重场景碎片化以及多个不同对象同时出现相互支持的异常。\n"
    "\n"
    "不能将所有异常特征机械相加。"
    "例如“人物不完整”“缺少手臂”“缺少部分身体结构”可能来自同一个结构缺失问题，"
    "不能当作三个完全独立的异常证据重复计算。\n"
    "\n"
    "同样，也不能因为一个对象结构正常就机械增加多个正常分数。"
    "应关注相互独立的证据来源。\n"
    "\n"
    "====================\n"
    "七、异常程度\n"
    "====================\n"
    "\n"
    "以下情况通常属于较强异常证据："
    "完全缺失房屋、树木或人物；人物明显不完整；房屋或人物结构严重怪异；"
    "整体画面极端微小或极度简化；对象之间极端分离；"
    "人物被异常重度涂黑；多个对象分别存在严重异常。\n"
    "\n"
    "以下情况通常只属于较弱辅助异常证据："
    "轻微比例问题、普通单线四肢、少量阴影、少量尖锐树枝、"
    "普通二维房屋、表情中性、没有背景装饰、简单画法等。\n"
    "\n"
    "严重程度比是否出现某个特征本身更加重要。"
    "例如轻微比例误差几乎不能作为异常依据，"
    "但极端、明显不可能的人体比例失调则可以成为较强异常证据。\n"
    "\n"
    "====================\n"
    "八、正常程度\n"
    "====================\n"
    "\n"
    "如果画面同时表现出以下多个特征，应积极增加判断为“心理状态正常”的权重：\n"
    "1. 房屋、树木、人物全部存在且基本完整；\n"
    "2. 三者大小和位置基本协调；\n"
    "3. 房屋结构稳定，主要组成部分合理；\n"
    "4. 树木结构自然，没有严重截断或畸形；\n"
    "5. 人物身体主要结构完整，比例不存在严重失调；\n"
    "6. 整体构图稳定且场景连贯；\n"
    "7. 线条自然、连续，没有异常大面积涂黑；\n"
    "8. 没有明显的选择性结构缺失；\n"
    "9. 即使绘画较简单，其简化方式在整幅画中保持一致；\n"
    "10. 仅存在少数能够由绘画水平、年龄、风格或普通构图解释的轻微异常。\n"
    "\n"
    "当上述积极特征在多个对象和整体构图中同时出现时，"
    "应把它们视为真实的正常支持证据，而不能因为发现一个轻微异常点就忽略这些积极特征。\n"
    "\n"
    "====================\n"
    "九、最终判定规则\n"
    "====================\n"
    "\n"
    "更倾向于判断为“存在心理异常倾向”的情况包括：\n"
    "1. 出现一个非常严重的异常，同时还有其他独立异常证据支持；\n"
    "2. 房屋、树木、人物中的多个对象分别出现明显异常；\n"
    "3. 元素缺失、结构扭曲、极端小型化、异常强调、空间隔离等不同类型异常同时出现；\n"
    "4. 整体构图异常与具体对象异常方向一致；\n"
    "5. 出现无法由整体绘画能力解释的明显选择性异常，例如房屋和树木较详细，"
    "但人物选择性缺少脸部或主要肢体。\n"
    "\n"
    "更倾向于判断为“心理状态正常”的情况包括：\n"
    "1. 房屋、树木和人物都能够识别并基本完整；\n"
    "2. 三个对象及其内部结构比例大致合理；\n"
    "3. 对象之间空间关系自然，整体场景具有连贯性；\n"
    "4. 没有严重结构缺失或严重结构畸形；\n"
    "5. 没有明显异常的大面积涂黑、极端小型化或异常隔离；\n"
    "6. 画面存在多个相互独立的正常积极特征；\n"
    "7. 只有一个或两个轻微、模糊或者能够由普通绘画习惯解释的异常点。\n"
    "\n"
    "如果正常证据明显多于并强于异常证据，应选择“心理状态正常”。"
    "如果异常证据在严重程度、独立性和一致性上明显强于正常证据，应选择“存在心理异常倾向”。"
    "如果两边证据接近，应重点比较最可靠、最明显、最独立的证据，"
    "不能因为Prompt中列出的异常特征较多而偏向异常，也不能默认正常是安全答案。\n"
    "\n"
    "不要把“发现异常特征”理解成“必须判断异常”。"
    "普通人的绘画同样可能存在比例不准确、线条简单、缺少装饰、单线四肢、透视错误等现象。"
    "只有异常程度明显，或者多个独立异常证据相互支持时，才应明显提高异常标签的判断权重。\n"
    "\n"
    "作品不需要漂亮、复杂、写实或具有专业绘画技巧才能属于心理状态正常。"
    "低绘画水平本身不是心理异常证据。\n"
)

# 第一阶段：只让它列证据，不给结论。刻意不提"选项"两个字，
# 避免第一轮就把答案定死，后面那轮变成照抄。
ANALYZE_SUFFIX = (
    "\n====================\n"
    "本轮任务\n"
    "====================\n\n"
    "请按上面的顺序逐项检查这幅画，分成两组分别写出你**实际观察到**的证据：\n"
    "【支持正常的证据】……\n"
    "【支持异常的证据】……\n"
    "每条都要指明对应的具体画面内容，例如哪个对象的哪个部位。"
    "只写真正看得到的，看不清或不确定的就不要写，宁可少写。"
    "总长控制在 400 字以内。\n"
    "这一轮不要给出任何最终判断，也不要写倾向性的总结句。"
)

# 第二阶段：图 + 上一轮证据 -> 单字母答案
DECIDE_PROMPT = (
    "现在结合这幅画和你上面列出的两组证据，比较双方的严重程度、独立性和一致性，"
    "给出最终判断。\n"
    f"A = {CHOICES[0]}\n"
    f"B = {CHOICES[1]}\n"
    "只输出一个大写字母 A 或 B，不要输出任何其他内容。"
)

# 单阶段模式用的结尾，保留原来的行为方便对照
DIRECT_SUFFIX = (
    "\n你可以在内部完成上述分析，但最终只能输出一个大写字母。\n"
    f"A = {CHOICES[0]}\n"
    f"B = {CHOICES[1]}\n"
    "不要输出分析过程、理由、特征、概率、置信度或任何其他文字。"
)

# ---------------------------------------------------------------------------
# extract 模式：只问客观事实，一个心理学词都不出现。
# think 模式失败的原因是把评分规则提前给了模型，它直接照抄清单，
# 所以这里的 schema 全部是"画上有没有 / 有几个 / 多大"这种能用眼睛回答的问题。
# ---------------------------------------------------------------------------
YN = {"enum": ["yes", "no", "unclear"]}
YNN = {"enum": ["yes", "no", "none", "unclear"]}          # none = 该对象根本没画
SIZE = {"enum": ["tiny", "small", "medium", "large", "none"]}
RATIO = {"enum": ["much_smaller", "smaller", "proportionate",
                  "larger", "much_larger", "none", "unclear"]}
AMOUNT = {"enum": ["none", "few", "moderate", "many", "unclear"]}
LEVEL = {"enum": ["none", "slight", "obvious", "unclear"]}
LEN = {"enum": ["very_short", "short", "proportionate", "long", "very_long",
                "none", "unclear"]}

# 上一版 40 个字段问的全是"有没有"，信息上限只有 0.639。
# 这一版改成问"多大、什么形状、怎么连接、线条什么样"——
# 也就是 HTP 量表实际打分的维度。
FEATURE_SCHEMA_PROPS = {
    # ---- 整体构图 ----
    "content_area_ratio": {"enum": ["under_10", "10_25", "25_50", "50_75",
                                    "over_75", "unclear"]},
    "content_position_h": {"enum": ["far_left", "left", "center", "right",
                                    "far_right", "unclear"]},
    "content_position_v": {"enum": ["top", "upper", "middle", "lower",
                                    "bottom", "unclear"]},
    "composition_balance": {"enum": ["balanced", "left_heavy", "right_heavy",
                                     "top_heavy", "bottom_heavy", "unclear"]},
    "ground_line_type": {"enum": ["none", "straight", "wavy", "hills",
                                  "multiple", "unclear"]},
    "objects_share_ground": YN,
    "objects_overlap": YN,
    "border_drawn": YN,
    "unfinished_elements": {"enum": ["none", "one", "multiple", "unclear"]},
    "overall_detail_level": {"enum": ["minimal", "low", "moderate", "high"]},
    "simplification_consistent": YN,
    "largest_object": {"enum": ["house", "tree", "person", "other", "unclear"]},
    "distance_house_tree": {"enum": ["touching", "close", "moderate", "far",
                                     "none", "unclear"]},
    "distance_person_house": {"enum": ["touching", "close", "moderate", "far",
                                       "none", "unclear"]},
    "distance_person_tree": {"enum": ["touching", "close", "moderate", "far",
                                      "none", "unclear"]},

    # ---- 线条与笔触（量表权重很高，上一版完全没有）----
    "line_continuity": {"enum": ["continuous", "mostly_continuous",
                                 "frequently_broken", "sketchy", "unclear"]},
    "line_repetition": {"enum": ["none", "occasional", "frequent", "unclear"]},
    "line_tremor": LEVEL,
    "line_pressure_variation": {"enum": ["uniform", "moderate",
                                         "highly_variable", "unclear"]},
    "line_thickness": {"enum": ["very_thin", "thin", "medium", "thick",
                                "unclear"]},
    "erasure_marks": AMOUNT,
    "overdrawn_areas": {"enum": ["none", "one", "multiple", "unclear"]},
    "shading_amount": {"enum": ["none", "light", "moderate", "heavy"]},
    "shading_location": {"enum": ["none", "house", "tree", "person",
                                  "background", "multiple", "unclear"]},
    "scribble_present": YN,
    "large_dark_areas": YN,

    # ---- 房屋 ----
    "house_present": YN,
    "house_size_ratio": SIZE,
    "house_wall_shape": {"enum": ["rectangle", "trapezoid", "irregular",
                                  "none", "unclear"]},
    "house_walls_upright": {"enum": ["upright", "slightly_tilted",
                                     "strongly_tilted", "none", "unclear"]},
    "house_roof_present": YNN,
    "house_roof_ratio": RATIO,
    "house_roof_attached": {"enum": ["attached", "partially_detached",
                                     "detached", "none", "unclear"]},
    "house_door_present": YNN,
    "house_door_ratio": RATIO,
    "house_door_position": {"enum": ["center", "left", "right", "none",
                                     "unclear"]},
    "house_door_open": {"enum": ["open", "closed", "none", "unclear"]},
    "house_window_count": {"enum": ["0", "1", "2", "3", "4+", "none",
                                    "unclear"]},
    "house_window_ratio": RATIO,
    "house_window_grid": YNN,
    "house_chimney": YNN,
    "house_smoke": {"enum": ["none", "thin", "thick", "unclear"]},
    "house_path_to_door": YNN,
    "house_detail_amount": AMOUNT,

    # ---- 树 ----
    "tree_present": YN,
    "tree_size_ratio": SIZE,
    "tree_trunk_width": {"enum": ["very_thin", "thin", "moderate", "thick",
                                  "none", "unclear"]},
    "tree_trunk_shape": {"enum": ["straight", "curved", "leaning", "broken",
                                  "none", "unclear"]},
    "tree_trunk_cut_off": YNN,
    "tree_trunk_scars": YNN,
    "tree_crown_present": YNN,
    "tree_crown_shape": {"enum": ["round", "oval", "irregular", "flattened",
                                  "scribbled", "none", "unclear"]},
    "tree_crown_ratio": RATIO,
    "tree_branches_direction": {"enum": ["none", "upward", "outward",
                                         "downward", "mixed", "unclear"]},
    "tree_branch_tips": {"enum": ["rounded", "open_ended", "pointed",
                                  "none", "unclear"]},
    "tree_branches_broken": YNN,
    "tree_leaves_amount": {"enum": ["none", "sparse", "moderate", "dense",
                                    "unclear"]},
    "tree_fruit": YNN,
    "tree_roots": {"enum": ["none", "simple", "elaborate", "unclear"]},
    "tree_dead_appearance": YNN,

    # ---- 人物 ----
    "person_present": YN,
    "person_count": {"enum": ["0", "1", "2", "3+"]},
    "person_size_ratio": SIZE,
    "person_style": {"enum": ["stick_figure", "outline", "detailed",
                              "none", "unclear"]},
    "person_head_present": YNN,
    "person_head_body_ratio": RATIO,
    "person_neck": YNN,
    "person_facial_feature_count": {"enum": ["0", "1", "2", "3", "4+",
                                             "none", "unclear"]},
    "person_eyes_type": {"enum": ["none", "dots", "circles", "with_pupils",
                                  "closed", "unclear"]},
    "person_mouth_shape": {"enum": ["none", "up", "flat", "down", "open",
                                    "teeth", "unclear"]},
    "person_ears": YNN,
    "person_hair_amount": {"enum": ["none", "sparse", "full", "unclear"]},
    "person_body_shape": {"enum": ["none", "single_line", "rectangle",
                                   "rounded", "detailed", "unclear"]},
    "person_arms_count": {"enum": ["0", "1", "2", "none", "unclear"]},
    "person_arm_length": LEN,
    "person_arm_attachment": {"enum": ["natural", "wrong_place", "detached",
                                       "none", "unclear"]},
    "person_arm_direction": {"enum": ["down", "outward", "up", "behind",
                                      "none", "unclear"]},
    "person_hands_detail": {"enum": ["none", "stub", "simple", "fingers",
                                     "fist", "unclear"]},
    "person_legs_count": {"enum": ["0", "1", "2", "none", "unclear"]},
    "person_leg_length": LEN,
    "person_feet_detail": {"enum": ["none", "stub", "simple", "shoes",
                                    "unclear"]},
    "person_symmetry": {"enum": ["symmetric", "slightly_asymmetric",
                                 "strongly_asymmetric", "none", "unclear"]},
    "person_parts_missing": {"enum": ["none", "one", "two", "three_plus",
                                      "unclear"]},
    "person_parts_detached": {"enum": ["none", "one", "multiple", "unclear"]},
    "person_clothing_detail": AMOUNT,
    "person_facing": {"enum": ["front", "side", "back", "none", "unclear"]},
    "person_on_ground": {"enum": ["on_ground", "floating", "none", "unclear"]},
    "person_heavily_shaded": YNN,

    # ---- 环境元素 ----
    "sun": YN,
    "clouds": YN,
    "rain": YN,
    "grass_or_flowers": YN,
    "path_or_road": YN,
    "water": YN,
    "fence": YN,
    "mountains": YN,
    "animals": YN,
    "birds": YN,
    "vehicles": YN,
    "text_present": YN,
    "color_used": YN,
}

FEATURE_SCHEMA = {
    "type": "object",
    "properties": FEATURE_SCHEMA_PROPS,
    "required": list(FEATURE_SCHEMA_PROPS),
    "additionalProperties": False,
}

EXTRACT_PROMPT = (
    "这是一张手绘图画。请只描述你在画面上**实际看到**的内容。\n"
    "\n"
    "总则：\n"
    "1. 只回答看得见的事实，不要做任何解读、评价或推测。\n"
    "2. 不要判断画得好不好、像不像、正常不正常。\n"
    "3. 看不清或没有把握填 unclear；该对象根本没画填 none。这两个不要混。\n"
    "4. 画中有多个人时，person 相关字段按其中画得最完整的那一个回答。\n"
    "\n"
    "字段口径：\n"
    "\n"
    "【面积与位置】\n"
    "content_area_ratio：所有笔画的外接范围占整张纸的比例，按百分比档位选。\n"
    "content_position_h / _v：画面内容的重心落在纸的哪个区域，各分五档。\n"
    "*_size_ratio：该对象占整张纸的面积。tiny 不到 5%，small 5%~15%，"
    "medium 15%~40%，large 超过 40%；没画该对象填 none。\n"
    "distance_*：两个对象边缘之间的距离。touching 相接或重叠，"
    "close 小于较小对象的宽度，moderate 一到三倍，far 超过三倍。\n"
    "\n"
    "【比例类，统一口径】\n"
    "house_roof_ratio：屋顶高度相对墙体高度。\n"
    "house_door_ratio：门的面积相对整面墙。\n"
    "house_window_ratio：单个窗户面积相对整面墙。\n"
    "tree_crown_ratio：树冠面积相对树干长度所暗示的合理树冠。\n"
    "person_head_body_ratio：头部相对躯干。\n"
    "以上一律选 much_smaller / smaller / proportionate / larger / much_larger，"
    "明显偏离常识比例才用 much_ 那两档。\n"
    "person_arm_length / person_leg_length：四肢长度相对躯干，"
    "very_short 到 very_long 五档。\n"
    "\n"
    "【线条与笔触】\n"
    "line_continuity：线条是一笔到底还是断断续续。\n"
    "line_repetition：同一条线被来回描了几遍。none 没有，occasional 个别地方，"
    "frequent 多处反复。\n"
    "line_tremor：线条有没有抖动、不稳的痕迹。\n"
    "line_pressure_variation：同一幅画里线条深浅粗细的变化程度。\n"
    "erasure_marks：擦改留下的痕迹、残影或纸面起毛。\n"
    "overdrawn_areas：局部被明显反复加重、涂实的区域个数。\n"
    "shading_amount：涂抹或排线阴影的总量。\n"
    "large_dark_areas：是否存在成片涂满的深色区域，面积超过整幅画的 5%。\n"
    "\n"
    "【结构与连接】\n"
    "house_roof_attached：屋顶和墙体是否接合。\n"
    "person_arm_attachment：手臂接在躯干上的位置是否正常，"
    "wrong_place 指接在头上或腰下等明显错位，detached 指与躯干分离。\n"
    "person_parts_missing：头、躯干、双臂、双腿、手、脚里缺了几项，"
    "能由姿势或遮挡解释的不算。\n"
    "person_parts_detached：身体部件之间明显断开的处数。\n"
    "person_symmetry：左右两侧的对应部件在长度和位置上是否对称。\n"
    "tree_trunk_cut_off：树干在画面边缘被截断，或者顶端突然平切。\n"
    "\n"
    "严格按给定的 JSON 结构输出，不要输出任何其他文字。"
)


def encode_image(path, max_side=1024, normalize=False):
    """normalize=True 时抹掉采集差异：长边一律缩放到 max_side（不管原图多大），
    并做逐图对比度拉伸。诊断发现 college 域的标签能被灰度中位数预测到 AUC 0.815，
    不做这一步，模型读到的是扫描仪设置而不是画。"""
    img = Image.open(path).convert("RGB")
    if normalize:
        scale = max_side / max(img.size)
        img = img.resize((max(int(img.width * scale), 1),
                          max(int(img.height * scale), 1)), Image.LANCZOS)
        g = np.asarray(img.convert("L"), dtype=np.float32)
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        if hi - lo < 1e-3:
            hi = lo + 1.0
        g = np.clip((g - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
        img = Image.fromarray(g).convert("RGB")
    elif max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def list_samples(data_dir):
    samples = []
    for env in sorted(os.listdir(data_dir)):
        env_dir = os.path.join(data_dir, env)
        if not os.path.isdir(env_dir):
            continue
        for cls in CLASS_DIRS:
            cls_dir = os.path.join(env_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                samples.append((os.path.join(cls_dir, fname), env, cls))
    return samples


def user_msg(b64, text):
    return {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": text},
    ]}


def choose(client, model, messages):
    """引导解码出 A/B，同时取两个字母的 logprob 换算成 P(异常)。"""
    resp = client.chat.completions.create(
        model=model, messages=messages,
        temperature=0, max_tokens=4,
        logprobs=True, top_logprobs=5,
        extra_body={"guided_choice": LETTERS},
    )
    pred = LETTER2DIR.get(resp.choices[0].message.content.strip())
    score = None
    try:
        lp = {t.token.strip(): t.logprob
              for t in resp.choices[0].logprobs.content[0].top_logprobs}
        if "A" in lp and "B" in lp:
            m = max(lp["A"], lp["B"])
            ea, eb = math.exp(lp["A"] - m), math.exp(lp["B"] - m)
            score = ea / (ea + eb)          # P(00 = 存在心理异常倾向)
    except Exception:
        pass                                 # 服务端不返回 logprobs 就只留标签
    return pred, score


# ---------------------------------------------------------------------------
# Responses API 后端：没有引导解码，也没有 logprobs，
# 所以 schema 用文字描述，标签和风险分让模型以 JSON 输出，客户端校验。
# ---------------------------------------------------------------------------
def schema_as_text():
    lines = []
    for k, spec in FEATURE_SCHEMA_PROPS.items():
        lines.append(f"  \"{k}\": {' | '.join(spec['enum'])}")
    return "{\n" + ",\n".join(lines) + "\n}"


def clean_features(raw):
    """把模型返回的 JSON 卡回 schema，越界的值一律记为 unclear。"""
    out = {}
    for k, spec in FEATURE_SCHEMA_PROPS.items():
        allowed = spec["enum"]
        v = str(raw.get(k, "")).strip().lower()
        if v in allowed:
            out[k] = v
        else:
            out[k] = "unclear" if "unclear" in allowed else allowed[-1]
    return out


API_JUDGE_SUFFIX = (
    "\n只输出如下 JSON，不要输出任何其他文字：\n"
    '{"label": "A" 或 "B", "risk": 0 到 100 的整数}\n'
    f"A = {CHOICES[0]}，B = {CHOICES[1]}。\n"
    "risk 表示你认为该作品属于「" + CHOICES[0] + "」的可能性，"
    "0 表示几乎不可能，100 表示几乎确定。请充分使用整个区间，不要只给 0、50、100。"
)


def parse_judge(text):
    from api_backend import loads_lenient
    d = loads_lenient(text)
    label = str(d.get("label", "")).strip().upper()[:1]
    pred = LETTER2DIR.get(label)
    score = None
    try:
        score = min(max(float(d["risk"]), 0.0), 100.0) / 100.0
    except (KeyError, TypeError, ValueError):
        pass
    return pred, score


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"enum": LETTERS},
        "risk": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["label", "risk"],
    "additionalProperties": False,
}


def predict_api(client, path, mode, max_tokens, normalize):
    from api_backend import loads_lenient
    b64 = encode_image(path, normalize=normalize)
    # llama.cpp 支持 json_schema 硬约束；Responses 后端不支持，schema 会被忽略，
    # 退回 json_object + prompt 里的文字描述，靠 clean_features 兜底。
    schema_ok = hasattr(client, "call") and "schema" in \
        client.call.__code__.co_varnames

    def call(text, schema=None, **kw):
        if schema is not None and schema_ok:
            return client.call(text, schema=schema, **kw)
        return client.call(text, json_mode=schema is not None, **kw)

    if mode == "extract":
        text = (EXTRACT_PROMPT + "\n\n严格按下面的结构输出，"
                "每个字段只能取给定的值之一：\n" + schema_as_text())
        out = call(text, schema=FEATURE_SCHEMA, b64=b64,
                   max_output_tokens=max_tokens)
        return None, None, json.dumps(clean_features(loads_lenient(out)),
                                      ensure_ascii=False)

    if mode == "direct":
        pred, score = parse_judge(call(
            RULES + API_JUDGE_SUFFIX, schema=JUDGE_SCHEMA, b64=b64,
            max_output_tokens=400))
        return pred, score, ""

    analysis = client.call(RULES + ANALYZE_SUFFIX, b64=b64,
                           max_output_tokens=max_tokens)
    pred, score = parse_judge(call(
        "下面是你对这幅画的观察记录：\n\n" + (analysis or "").strip() +
        "\n\n" + DECIDE_PROMPT + API_JUDGE_SUFFIX,
        schema=JUDGE_SCHEMA, b64=b64, max_output_tokens=400))
    return pred, score, analysis


def extract(client, model, b64, max_tokens):
    """引导解码出固定 schema 的 JSON，字段和取值都由 schema 保证，不用写解析。"""
    resp = client.chat.completions.create(
        model=model, messages=[user_msg(b64, EXTRACT_PROMPT)],
        temperature=0, max_tokens=max_tokens,
        extra_body={"guided_json": FEATURE_SCHEMA},
    )
    return json.loads(resp.choices[0].message.content)


def predict(client, model, path, mode="think", max_analysis_tokens=600,
            normalize=False):
    b64 = encode_image(path, normalize=normalize)
    if mode == "extract":
        feat = extract(client, model, b64, max_analysis_tokens)
        return None, None, json.dumps(feat, ensure_ascii=False)

    if mode == "direct":
        pred, score = choose(client, model, [user_msg(b64, RULES + DIRECT_SUFFIX)])
        return pred, score, ""

    img_msg = user_msg(b64, RULES + ANALYZE_SUFFIX)
    r1 = client.chat.completions.create(
        model=model, messages=[img_msg],
        temperature=0, max_tokens=max_analysis_tokens)
    analysis = (r1.choices[0].message.content or "").strip()

    # 第二轮复用同一段前缀，vLLM 的 prefix caching 会命中，图不用重算
    pred, score = choose(client, model, [
        img_msg,
        {"role": "assistant", "content": analysis},
        {"role": "user", "content": DECIDE_PROMPT},
    ])
    return pred, score, analysis


def auc_score(pairs):
    """pairs = [(score, is_positive)]，正类为 00（存在心理异常倾向）。

    用秩和公式算 AUC，并列分数取平均秩，等价于 Mann-Whitney U。
    """
    pairs = [(s, y) for s, y in pairs if s is not None]
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    pairs.sort(key=lambda t: t[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def balanced_acc(pairs, thr):
    """pairs = [(score, is_positive)]，score >= thr 判为正类（00 存在异常）。"""
    tp = sum(1 for s, y in pairs if y and s >= thr)
    fn = sum(1 for s, y in pairs if y and s < thr)
    fp = sum(1 for s, y in pairs if not y and s >= thr)
    tn = sum(1 for s, y in pairs if not y and s < thr)
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return (tpr + tnr) / 2


def best_threshold(pairs):
    cands = sorted({s for s, _ in pairs})
    if not cands:
        return None
    return max(cands, key=lambda t: balanced_acc(pairs, t))


def calibrated_report(rows):
    """留一域校准：阈值只在另外两个域上选，用到留出域，避免拿测试集调阈值。"""
    usable = [r for r in rows if r[4] is not None]
    if len(usable) < len(rows) * 0.9:
        return
    envs = sorted({r[1] for r in usable})
    if len(envs) < 2:
        return
    print("\n留一域阈值校准（阈值在另外两个域上按平衡准确率选出）")
    print(f"{'held-out':10s} {'thr':>7s} {'acc':>8s} {'balanced':>9s}")
    accs, bals = [], []
    for env in envs:
        tr = [(r[4], r[2] == "00") for r in usable if r[1] != env]
        te = [(r[4], r[2] == "00") for r in usable if r[1] == env]
        thr = best_threshold(tr)
        acc = sum(1 for s, y in te if (s >= thr) == y) / len(te)
        bal = balanced_acc(te, thr)
        accs.append(acc)
        bals.append(bal)
        print(f"{ENV_NAME.get(env, env):10s} {thr:7.3f} {acc:8.4f} {bal:9.4f}")
    print(f"{'mean':10s} {'':7s} {sum(accs) / len(accs):8.4f} "
          f"{sum(bals) / len(bals):9.4f}")


def report(rows):
    env_hit, env_tot = collections.Counter(), collections.Counter()
    cls_hit, cls_tot = collections.Counter(), collections.Counter()
    cm = collections.Counter()
    n_bad = 0
    for r in rows:
        env, gt, pred = r[1], r[2], r[3]
        env_tot[env] += 1
        cls_tot[(env, gt)] += 1
        if pred == gt:
            env_hit[env] += 1
            cls_hit[(env, gt)] += 1
        if pred is None:
            n_bad += 1
        cm[(gt, pred)] += 1

    print(f"{'env':8s} {'n':>5s} {'acc':>8s} {'balanced':>9s} {'AUC':>7s} {'base':>7s}")
    for env in sorted(env_tot):
        rec = [cls_hit[(env, c)] / max(cls_tot[(env, c)], 1) for c in CLASS_DIRS]
        a = auc_score([(r[4], r[2] == "00") for r in rows if r[1] == env])
        base = max(cls_tot[(env, c)] for c in CLASS_DIRS) / env_tot[env]
        print(f"{ENV_NAME.get(env, env):8s} {env_tot[env]:5d} "
              f"{env_hit[env] / env_tot[env]:8.4f} {sum(rec) / len(rec):9.4f} "
              f"{a if a is None else round(a, 4)!s:>7s} {base:7.4f}")

    n = sum(env_tot.values())
    rec = [sum(cls_hit[(e, c)] for e in env_tot) / max(sum(cls_tot[(e, c)] for e in env_tot), 1)
           for c in CLASS_DIRS]
    a = auc_score([(r[4], r[2] == "00") for r in rows])
    base = max(sum(cls_tot[(e, c)] for e in env_tot) for c in CLASS_DIRS) / n
    print(f"{'ALL':8s} {n:5d} {sum(env_hit.values()) / n:8.4f} "
          f"{sum(rec) / len(rec):9.4f} {a if a is None else round(a, 4)!s:>7s} {base:7.4f}")
    print(f"failed={n_bad}")
    for gt in CLASS_DIRS:
        row = {p: cm[(gt, p)] for p in CLASS_DIRS + [None] if cm[(gt, p)]}
        print(f"  真实 {gt} ({CLASS_DESC[gt]}) -> {row}")

    scores = [r[4] for r in rows if r[4] is not None]
    if scores:
        qs = sorted(scores)
        pick = [qs[int(len(qs) * p)] for p in (0.0, 0.25, 0.5, 0.75, 0.99)]
        print(f"score 分布 min/p25/p50/p75/p99 = "
              f"{' '.join(f'{v:.3f}' for v in pick)}，不同取值 {len(set(scores))} 个")
    calibrated_report(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/HTP")
    # 多卡时每张卡起一个独立实例，这里用逗号分隔多个地址，请求轮流发
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default="qwen-vl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="output/qwen_htp.csv")
    # think = 先让模型写证据再判断；direct = 老的一次调用；
    # extract = 只抽客观视觉特征，交给 htp_probe.py 训分类器
    ap.add_argument("--mode", choices=["think", "direct", "extract"], default="think")
    # extract 要输出 40 个字段的 JSON，比写分析更费 token
    ap.add_argument("--max-analysis-tokens", type=int)
    # 只跑前 N 张，用来快速验证 prompt 和链路
    ap.add_argument("--limit", type=int)
    # 只想用已有结果重算指标时用这个，不再发请求
    ap.add_argument("--from-csv")
    # vllm = 本地服务；responses = 外部 OpenAI Responses 兼容 API
    ap.add_argument("--backend", choices=["vllm", "responses", "llamacpp"],
                    default="vllm")
    # 抹掉扫描亮度和分辨率差异，避免模型读到采集痕迹而不是画面
    ap.add_argument("--normalize", action="store_true")
    # Qwen3.5/3.6 的思考模式。开了要把 --max-analysis-tokens 一起调大，
    # 否则 token 全烧在推理里，content 返回空
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()
    if args.max_analysis_tokens is None:
        args.max_analysis_tokens = 3000 if args.mode == "extract" else 600

    if args.from_csv:
        with open(args.from_csv, encoding="utf-8") as f:
            rows = [(r["path"], r["env"], r["label"], r["pred"] or None,
                     float(r["score"]) if r.get("score") else None)
                    for r in csv.DictReader(f)]
        print(f"{len(rows)} rows from {args.from_csv}")
        report(rows)
        return

    if args.backend == "responses":
        from api_backend import ResponsesClient
        clients = [ResponsesClient(u.strip(), args.model)
                   for u in args.base_url.split(",")]
    elif args.backend == "llamacpp":
        from api_backend import ChatClient
        clients = [ChatClient(u.strip(), args.model,
                              enable_thinking=args.thinking)
                   for u in args.base_url.split(",")]
        if args.thinking and args.max_analysis_tokens < 3000:
            print(f"[warn] --thinking 开着但 max-analysis-tokens 只有 "
                  f"{args.max_analysis_tokens}，推理很可能吃满导致 content 为空，"
                  f"建议 4000 以上")
    else:
        clients = [OpenAI(base_url=u.strip(), api_key=args.api_key)
                   for u in args.base_url.split(",")]
    samples = list_samples(args.data_dir)
    if args.limit:
        samples = samples[::max(len(samples) // args.limit, 1)][:args.limit]
    print(f"{len(samples)} images, {len(clients)} endpoint(s), mode={args.mode}")
    counter = itertools.count(1)

    def work(idx_item):
        idx, (path, env, gt) = idx_item
        client = clients[idx % len(clients)]
        try:
            if args.backend in ("responses", "llamacpp"):
                pred, score, analysis = predict_api(
                    client, path, args.mode, args.max_analysis_tokens,
                    args.normalize)
            else:
                pred, score, analysis = predict(
                    client, args.model, path, args.mode,
                    args.max_analysis_tokens, args.normalize)
        except Exception as e:  # 单张失败不影响整体，记为 None
            print(f"[fail] {path}: {e}")
            pred, score, analysis = None, None, ""
        if args.mode == "extract":
            print(f"[{next(counter)}/{len(samples)}] {path} "
                  f"{'ok' if analysis else 'FAIL'}")
        else:
            print(f"[{next(counter)}/{len(samples)}] {path} -> {pred} "
                  f"p(异常)={'-' if score is None else round(score, 3)}")
        return path, env, gt, pred, score, analysis

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(work, enumerate(samples)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    col = "features" if args.mode == "extract" else "analysis"
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "env", "label", "pred", "score", col])
        w.writerows(rows)

    if args.mode == "extract":
        ok = sum(1 for r in rows if r[5])
        print(f"\n抽取完成 {ok}/{len(rows)}，写入 {args.out}")
        print(f"接着跑：python htp_probe.py {args.out}")
        return
    report(rows)


if __name__ == "__main__":
    main()