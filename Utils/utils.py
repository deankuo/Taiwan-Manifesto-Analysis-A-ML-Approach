# Import packages
import pandas as pd

def load_dataframe(df, year):
    df["CONTENT"]= df["CONTENT"].astype(str)
    if year in set([2004, 2008]):
        # Groupby and combine by LNAME
        df = df.groupby('LNAME').agg({'TH': 'first',
                                      'AREA': 'first',
                                      'BYELE': 'first',
                                      'PARTY': 'first',
                                      'ELE': 'first',
                                      'CONTENT': ' + '.join})
    else:
        df = df.groupby('LNAME').agg({'TH': 'first',
                                      'AREA': 'first',
                                      'BYELE': 'first',
                                      'PARTY': 'first',
                                      'ELE': 'first',
                                      'CONTENT': lambda x: ' + '.join(x) if not all(pd.isnull(x)) else ''})
        
        # 原先我只有coding好當選的人為1
        df['ELE'] = df['ELE'].fillna(0)

    # 去掉pdf檔出現的多餘空格
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace(' ', ''))
    # 去掉換行和tab鍵，這邊不能把這兩行一起執行是因為有些候選人的選舉公報只用換行來切割句子
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace('\n', ' ').replace('\t', ' '))

    df[['AREA', 'PARTY', 'CONTENT']] = df[['AREA', 'PARTY', 'CONTENT']].applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.insert(4, 'ABORIGINE', df['AREA'].apply(lambda x: 1 if x[-3:] == '原住民' else 0))

    # Reset index
    df = df.reset_index()
    return df