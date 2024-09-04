import pandas as pd
from faker import Faker
import random

# 创建 Faker 实例
fake = Faker("zh_CN")

# 药品建议列表
medications = [
    ("氟西汀", "20mg/天"),
    ("舍曲林", "50mg/天"),
    ("帕罗西汀", "40mg/天"),
    ("文拉法辛", "75mg/天"),
    ("碳酸锂", "300mg/天"),
    ("阿莫西林", "250mg/天"),
    ("利他林", "10mg/天"),
    ("地西泮", "10mg/天"),
    ("氯氮平", "25mg/天")
]

# 诊断结果列表
diagnoses = [
    "广泛性焦虑障碍",
    "抑郁症",
    "强迫症",
    "社交恐惧症",
    "双相情感障碍",
    "创伤后应激障碍",
    "注意力缺陷障碍",
    "失眠症",
    "恐慌症"
]

# 生成数据
data = []
names = set()
while len(data) < 100:
    name = fake.name()
    if name in names:
        continue
    names.add(name)

    gender = random.choice(['男', '女'])
    age = random.randint(20, 50)
    chief_complaint = random.choice(diagnoses)

    symptoms = {
        '广泛性焦虑障碍': '持续担忧、易怒、肌肉紧张',
        '抑郁症': '情绪低落、疲劳感、兴趣丧失',
        '强迫症': '重复行为、过度担忧、强迫思维',
        '社交恐惧症': '害怕社交场合、焦虑、逃避社交',
        '双相情感障碍': '情绪极度波动、躁狂与抑郁交替',
        '创伤后应激障碍': '创伤回忆、情绪麻木、警觉性增强',
        '注意力缺陷障碍': '注意力不集中、易分心、任务完成困难',
        '失眠症': '入睡困难、早醒、睡眠质量差',
        '恐慌症': '突发性强烈焦虑、心悸、呼吸急促'
    }

    symptom = symptoms[chief_complaint]
    medication = random.choice(medications)
    medication_str = f"{medication[0]} {medication[1]}"

    data.append([name, gender, age, chief_complaint, symptom, chief_complaint, medication_str])

# 创建 DataFrame
df = pd.DataFrame(data, columns=['姓名', '性别', '年龄', '患者主述', '症状', '诊断结果', '用药建议'])

# 保存为 Excel 文件
df.to_excel('患者信息.xlsx', index=False)
