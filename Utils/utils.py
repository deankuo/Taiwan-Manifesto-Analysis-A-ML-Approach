# Import packages
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from opencc import OpenCC

# list
years = [1992, 1995, 1998, 2001, 2004, 2008, 2012, 2016, 2020, 2024]

# Self defined functions
def load_dataframe(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Load the original raw data of manifesto."""
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
    # df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace('\n', ' ').replace('\t', ' '))

    df[['AREA', 'PARTY', 'CONTENT']] = df[['AREA', 'PARTY', 'CONTENT']].map(lambda x: x.strip() if isinstance(x, str) else x)
    df.insert(4, 'ABORIGINE', df['AREA'].apply(lambda x: 1 if x[-3:] == '原住民' else 0))

    # Reset index
    df = df.reset_index()
    return df

# Find duplicate names
def find_duplicate_lnames(dfs: dict):
    for name, df in dfs.items():
        if name in set([2004, 2008]):
            continue
        else:
            print(name)
            duplicate_lnames = df[df.duplicated('LNAME', keep=False)]['LNAME']
            if not duplicate_lnames.empty:
                print(f"In DataFrame '{name}', the following 'LNAME' values are duplicated:")
                print(duplicate_lnames.unique())

# Clean text
def text_clean(text: str) -> str: 
    """Delete all [] and () in 2004, 2008 data."""
    stop_word = "[]+- "
    result = ""
    for word in text:
        if word in stop_word:
            continue
        else:
            result += word
            
    result = re.sub("\([^)]*\)", "", result)
    return result

# Party encoding
def party_processing(df:pd.DataFrame) -> pd.DataFrame:
    """This function is used for party coding."""
    df['PARTY'] = df['PARTY'].replace({
        'KMT': '國民黨',
        'DPP': '民進黨',
        'Independent': '無',
        'independent': '無',
        'INDEPENDNET': '無',
        'INDEPENDENT': '無',
        'iNDEPENDENT': '無',
        '慧行黨': '臺灣慧行志工黨',
    })

    party_map = {'國民黨': 1, '民進黨': 2, '親民黨': 3, '新黨': 4, '無黨團結聯盟': 5, '台聯黨': 6, '無': 7, '台灣民眾黨': 8, '時代力量': 9, '台灣團結聯盟': 10}
    # 其他政黨
    party_codes = df['PARTY'].map(party_map).fillna(20)
    df.insert(df.columns.get_loc('PARTY') + 1, 'PARTY_CODE', party_codes)

    return df

# Content words sanity
def content_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Used for content processing, especially for replacing similar words."""
    df['LNAME'] = df['LNAME'].apply(lambda x: x.strip())
    df['AREA'].replace({'台': '臺'}, inplace=True, regex=True)
    df['CONTENT'] = df['CONTENT'].apply(text_clean)
    df['CONTENT'].replace({'台灣': '臺灣', 
                           '提昇': '提升',
                           '民衆': '民眾',
                           '啓動': '啟動'}, inplace=True, regex=True)
    return df

# Make sure the dataset is as clean as possible
def dataset_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """This function is used to make sure the configuration of datasets."""
    
    # Adjusts candidates' name
    df['LNAME'] = df.apply(lambda row: str(row['LNAME']) + '_' + str(int(row['TH'])), axis=1)
    
    # Double check if there are aboriginal candidates and by-election candidates left.
    df = df[~(((df['ABORIGINE'] == 1)) | (df['BYELE'] == 1))]
    
    # ELE
    if df.ELE.sum() < 73:
        df['ELE'] = np.where(df['當選註記'].isin(["1", 1, "*"]), 1, 0)
    
    # 現任
    df['現任'] = np.where(df['現任'].isin(['Y', 1, '是']), 1, 0)
    
    # 性別: Male: 1, Female: 0
    df['性別'] = df['性別'].apply(lambda x: 1 if x == 1 else 0) # 原始的女性coding有0, 2
    
    # Drop columns
    df = df.drop(['當選註記', 'ABORIGINE', 'BYELE'], axis=1)

    new_order = ['LNAME', 'TH', 'AREA', 'PARTY', 'PARTY_CODE', '性別', '現任', '學歷', 'ELE', '得票數', '得票率', 'CONTENT']
    new_names = {'性別': 'GENDER', '得票數':'VOTES', '得票率': 'VOTER_TURNOUT', '現任': 'INCUMBENT', '學歷': 'EDUCATION'}
    df = df[new_order].rename(columns=new_names)
    
    # Content
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.strip())
    
    # 去掉pdf檔出現的多餘空格
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace(' ', ''))
    
    # 把半形換成全形
    def half_to_full_width(s):
        n = []
        for char in s:
            if char == ',':
                n.append('，')
            else:
                code = ord(char)
                if code == 46:  # '.' 不動
                    pass
                elif 33 <= code <= 126:
                    code += 65248
                n.append(chr(code))
        return ''.join(n)

    df['CONTENT'] = df['CONTENT'].apply(half_to_full_width)
    
    cc = OpenCC('s2t') # 簡體到繁體
    def convert_to_traditional(text):
        return cc.convert(text)

    df['CONTENT'] = df['CONTENT'].apply(convert_to_traditional)  
    
    # 如果要以句子作為單位應該不能這樣做
    # 去掉tab鍵，這邊不能把這兩行一起執行是因為有些候選人的選舉公報只用換行來切割句子
    df['CONTENT'] = df['CONTENT'].apply(lambda x: x.replace('\t', ' '))
    
    # Ensure certain columns are integer type
    for column in ['TH', 'PARTY_CODE', 'GENDER', 'INCUMBENT', 'ELE', 'VOTES']:
        df[column] = df[column].astype(int)

    # Ensure certain columns are string type
    for column in ['LNAME', 'AREA', 'PARTY', 'CONTENT']:
        df[column] = df[column].astype(str)
    
    df['VOTER_TURNOUT'] = df['VOTER_TURNOUT'].astype(float)

    df = df.reset_index(drop=True)
    return df

# 計算候選人得票
def vote_calculate(group):
    """
        依照公職人員選舉罷免法第43條第1項規定：候選人除全國不分區及僑居國外國民立法委員選舉外，當選人在一人，得票數達各該選舉區當選票數三分之一以上者，
        當選人在二人以上，得票數達各該選舉區當選票數二分之一以上者，應補貼其競選費用，每票補貼新臺幣三十元。但其最高額，不得超過各該選舉區候選人競選經費最高金額。
    """
    least_win_votes = int(group.loc[group['ELE'] == 1, 'VOTES'].min())
    threshold = min(least_win_votes / (2 if group['ELE'].sum() > 2 else 3), 1000) # 以1000票作為最低門檻

    return group['VOTES'].apply(lambda x: x >= threshold)

# 主要政黨
def is_main_party(df: pd.DataFrame):
    """
    Main party coding.
    主要政黨有：國民黨、民進黨、親民黨、新黨、無黨團結聯盟、台聯黨、台灣民眾黨、時代力量、台灣團結聯盟
    """
    main_party = [1, 2, 3, 4, 5, 6, 8, 9, 10] 
    return df['PARTY_CODE'].isin(main_party)

# Visualization Functions
def party_plot(party_data: list, years: list, save=False):
    """This function is for plotting party number over years."""
    plt.figure(figsize=(8, 6))
    plt.plot(years, party_data, marker='o', linestyle='-', color='steelblue')

    for x, y in zip(years, party_data):
        plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10, color='black')

    plt.ylabel('The Amount of Party', fontsize=12)
    plt.title('Total Party Number', fontsize=14)
    plt.xticks(years, fontsize=10)
    plt.yticks(fontsize=10)
    
    if save:
        plt.savefig("./Graphs/Party Number.png")
    else:
        plt.show()
    

# Vote Datasets
def fill_education(dfs_vote: dict) -> dict:
    """This function is used for fill the missing value of 學歷 in vote_{year}.csv"""
    education_diff = []

    for year, df in dfs_vote.items():
        for index, row in df.iterrows():
            if row['學歷'] == '99' or row['學歷'] == 99 or pd.isnull(row['學歷']):
                for other_year, other_df in dfs_vote.items():
                    if other_year != year:
                        match = other_df[other_df['LNAME'] == row['LNAME']]
                        if not match.empty:
                            valid_education = match['學歷'][(match['學歷'] != '99') & (match['學歷'] != 99) & (~pd.isnull(match['學歷']))]
                            if not valid_education.empty:
                                updated_value = valid_education.iloc[0]
                                original_value = row['學歷']
                                if original_value not in ['99', 99, None] and updated_value != original_value:
                                    education_diff.append((row['LNAME'], year, other_year, original_value, updated_value))
                                dfs_vote[year].loc[index, '學歷'] = updated_value
                                break

    return dfs_vote, education_diff

def fill_age(dfs_vote: dict) -> dict:
    """This function is used for fill the missing value of 年齡 in vote_{year}.csv"""
    base_year = list(dfs_vote.keys())[0]

    for index, row in dfs_vote[base_year].iterrows():
        if row['年齡'] == 99 or pd.isnull(row['年齡']):
            for other_year, other_df in dfs_vote.items():
                if other_year != base_year:
                    match = other_df[other_df['LNAME'] == row['LNAME']]
                    if not match.empty:
                        valid_age = match['年齡'].iloc[0]
                        if valid_age != 99 and not pd.isnull(valid_age):
                            age_diff = int(other_year) - int(base_year)
                            updated_age = valid_age - age_diff
                            dfs_vote[base_year].loc[index, '年齡'] = updated_age
                            break

    return dfs_vote