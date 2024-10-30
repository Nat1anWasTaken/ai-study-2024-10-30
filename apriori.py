# 匯入套件
import pandas as pd
from apyori import apriori

df = pd.read_csv('data.csv', header=None)  # 讀入csv檔
print(df)  # 輸出試算表

# 資料預處理
data = df.values.tolist()
data = [[x for x in row if str(x) != 'nan'] for row in data]

# apriori函式運算
associations = apriori(
    data, min_support=0.01,
    min_confidence=0.4, min_lift=1.5
)
rules = list(associations)
print(rules[0])

# 輸出潛在知識
for rule in rules:
    for order_stat in rule.ordered_statistics:
        set_A = set(order_stat.items_base)
        set_B = set(order_stat.items_add)
        if len(set_A) == 0 or len(set_B) == 0:
            continue
        print(f'{set_A} => {set_B}')
        print(
            (f'Confidence: {order_stat.confidence :.4f}'
             f' Support: {rule.support :.4f}'
             f' Lift: {order_stat.lift :.4f}')
        )
