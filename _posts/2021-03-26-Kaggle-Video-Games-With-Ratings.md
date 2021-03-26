---
layout: post
title:  "Kaggle -Video Games With Ratings"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# Video Games With Ratings

- 데이터 셋 : https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings
- 게임이 가진 여러 변수를 통해 판매액을 분석하고자 하는 데이터 셋이다.
- 아웃라이어와 특정 게임이 가진 영향이 매우크기때문에 모델적용이 어려운데 모델을 만들어서 적용하기보다는 게임데이터에 관한 도메인지식과 양상을 습득하고자 한다.

## 라이브러리 설정 및 데이터 읽어들이기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

pd.set_option('display.max_columns', None)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>82.53</td>
      <td>76.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>322.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>40.24</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>Nintendo</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>35.52</td>
      <td>82.0</td>
      <td>73.0</td>
      <td>8.3</td>
      <td>709.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>32.77</td>
      <td>80.0</td>
      <td>73.0</td>
      <td>8</td>
      <td>192.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>Nintendo</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>31.37</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16719 entries, 0 to 16718
    Data columns (total 16 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Name             16717 non-null  object 
     1   Platform         16719 non-null  object 
     2   Year_of_Release  16450 non-null  float64
     3   Genre            16717 non-null  object 
     4   Publisher        16665 non-null  object 
     5   NA_Sales         16719 non-null  float64
     6   EU_Sales         16719 non-null  float64
     7   JP_Sales         16719 non-null  float64
     8   Other_Sales      16719 non-null  float64
     9   Global_Sales     16719 non-null  float64
     10  Critic_Score     8137 non-null   float64
     11  Critic_Count     8137 non-null   float64
     12  User_Score       10015 non-null  object 
     13  User_Count       7590 non-null   float64
     14  Developer        10096 non-null  object 
     15  Rating           9950 non-null   object 
    dtypes: float64(9), object(7)
    memory usage: 2.0+ MB


- Name : 게임의 이름
- Platform : 게임이 동작하는 콘솔
- Year_of_Release : 발매 년도
- Genre : 게임의 장르
- Publisher : 게임의 유통사
- NA_Sales : 북미 판매량 (Millions)
- EU_Sales : 유럽 연합 판매량 (Millions)
- JP_Sales : 일본 판매량 (Millions)
- Other_Sales : 기타 판매량 (아프리카, 일본 제외 아시아, 호주, EU 제외 유럽, 남미) (Millions)
- Global_Sales : 전세계 판매량
- Critic_Score : Metacritic 스태프 점수 (평균점수인듯 하다.)
- Critic_Count : Critic_Score에 사용된 점수의 수
- User_Score : Metacritic 구독자의 점수 (평균점수인듯 하다.)
- User_Count : User_Score에 사용된 점수의 수
- Developer : 게임의 개발사
- Rating: ESRB 등급 (19+, 17+, 등등)

## EDA 및 기초 통계 분석


```python
# 결손 데이터 삭제

df.isna().sum()
```




    Name                  2
    Platform              0
    Year_of_Release     269
    Genre                 2
    Publisher            54
    NA_Sales              0
    EU_Sales              0
    JP_Sales              0
    Other_Sales           0
    Global_Sales          0
    Critic_Score       8582
    Critic_Count       8582
    User_Score         6704
    User_Count         9129
    Developer          6623
    Rating             6769
    dtype: int64




```python
df.dropna(inplace=True)
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3840</th>
      <td>LEGO Jurassic World</td>
      <td>WiiU</td>
      <td>2015.0</td>
      <td>Action</td>
      <td>Warner Bros. Interactive Entertainment</td>
      <td>0.27</td>
      <td>0.19</td>
      <td>0.02</td>
      <td>0.04</td>
      <td>0.52</td>
      <td>71.0</td>
      <td>5.0</td>
      <td>8.1</td>
      <td>14.0</td>
      <td>TT Games</td>
      <td>E10+</td>
    </tr>
    <tr>
      <th>10027</th>
      <td>Disney Art Academy</td>
      <td>3DS</td>
      <td>2016.0</td>
      <td>Action</td>
      <td>Nintendo</td>
      <td>0.02</td>
      <td>0.01</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>72.0</td>
      <td>16.0</td>
      <td>6.8</td>
      <td>5.0</td>
      <td>Headstrong Games</td>
      <td>E</td>
    </tr>
    <tr>
      <th>10579</th>
      <td>The Whispered World</td>
      <td>PC</td>
      <td>2009.0</td>
      <td>Adventure</td>
      <td>Deep Silver</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.10</td>
      <td>70.0</td>
      <td>41.0</td>
      <td>7.6</td>
      <td>77.0</td>
      <td>Daedalic Entertainment</td>
      <td>E</td>
    </tr>
    <tr>
      <th>2397</th>
      <td>DiRT</td>
      <td>X360</td>
      <td>2007.0</td>
      <td>Racing</td>
      <td>Codemasters</td>
      <td>0.38</td>
      <td>0.40</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.87</td>
      <td>83.0</td>
      <td>59.0</td>
      <td>7.4</td>
      <td>94.0</td>
      <td>Codemasters</td>
      <td>E</td>
    </tr>
    <tr>
      <th>6605</th>
      <td>NASCAR 09</td>
      <td>PS3</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>Electronic Arts</td>
      <td>0.22</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.25</td>
      <td>65.0</td>
      <td>22.0</td>
      <td>7.5</td>
      <td>10.0</td>
      <td>EA Games</td>
      <td>E</td>
    </tr>
    <tr>
      <th>11070</th>
      <td>Last Rebellion</td>
      <td>PS3</td>
      <td>2010.0</td>
      <td>Role-Playing</td>
      <td>Nippon Ichi Software</td>
      <td>0.06</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.01</td>
      <td>0.09</td>
      <td>44.0</td>
      <td>30.0</td>
      <td>4.6</td>
      <td>34.0</td>
      <td>Hit Maker</td>
      <td>T</td>
    </tr>
    <tr>
      <th>11850</th>
      <td>I-Ninja</td>
      <td>XB</td>
      <td>2003.0</td>
      <td>Platform</td>
      <td>Namco Bandai Games</td>
      <td>0.05</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.07</td>
      <td>75.0</td>
      <td>18.0</td>
      <td>8.4</td>
      <td>8.0</td>
      <td>Argonaut Games</td>
      <td>T</td>
    </tr>
    <tr>
      <th>6100</th>
      <td>Shadow Hearts</td>
      <td>PS2</td>
      <td>2001.0</td>
      <td>Role-Playing</td>
      <td>Midway Games</td>
      <td>0.09</td>
      <td>0.07</td>
      <td>0.10</td>
      <td>0.02</td>
      <td>0.28</td>
      <td>73.0</td>
      <td>24.0</td>
      <td>8.8</td>
      <td>55.0</td>
      <td>Sacnoth</td>
      <td>M</td>
    </tr>
    <tr>
      <th>11918</th>
      <td>Serious Sam: Next Encounter</td>
      <td>PS2</td>
      <td>2004.0</td>
      <td>Shooter</td>
      <td>Global Star</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.07</td>
      <td>65.0</td>
      <td>27.0</td>
      <td>7.8</td>
      <td>17.0</td>
      <td>Climax Group</td>
      <td>M</td>
    </tr>
    <tr>
      <th>14040</th>
      <td>Legasista</td>
      <td>PS3</td>
      <td>2012.0</td>
      <td>Role-Playing</td>
      <td>Nippon Ichi Software</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.04</td>
      <td>68.0</td>
      <td>13.0</td>
      <td>6.3</td>
      <td>15.0</td>
      <td>System Prisma</td>
      <td>T</td>
    </tr>
  </tbody>
</table>

</div>



### 수치형 데이터 단순확인


```python
df.columns
```




    Index(['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales',
           'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score',
           'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'],
          dtype='object')




```python
sns.histplot(data=df, x='Year_of_Release', bins=16)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32bd8fa48>




![output_12_1](https://user-images.githubusercontent.com/77723966/112605156-563d7080-8e5a-11eb-8a5f-35d7df2e56bc.png)




```python
sns.rugplot(data=df, x='Global_Sales')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32c51fd88>




![output_13_1](https://user-images.githubusercontent.com/77723966/112605168-5b022480-8e5a-11eb-93fe-6748f2498e26.png)




```python
# 아웃라이어가 있는데 확인
df[df['Global_Sales'] > 60 ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>82.53</td>
      <td>76.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>322.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
  </tbody>
</table>

</div>




```python
df[df['Global_Sales'] > 30 ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year_of_Release</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
      <th>Critic_Score</th>
      <th>Critic_Count</th>
      <th>User_Score</th>
      <th>User_Count</th>
      <th>Developer</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.36</td>
      <td>28.96</td>
      <td>3.77</td>
      <td>8.45</td>
      <td>82.53</td>
      <td>76.0</td>
      <td>51.0</td>
      <td>8</td>
      <td>322.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>Nintendo</td>
      <td>15.68</td>
      <td>12.76</td>
      <td>3.79</td>
      <td>3.29</td>
      <td>35.52</td>
      <td>82.0</td>
      <td>73.0</td>
      <td>8.3</td>
      <td>709.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.61</td>
      <td>10.93</td>
      <td>3.28</td>
      <td>2.95</td>
      <td>32.77</td>
      <td>80.0</td>
      <td>73.0</td>
      <td>8</td>
      <td>192.0</td>
      <td>Nintendo</td>
      <td>E</td>
    </tr>
  </tbody>
</table>

</div>



- Wii Sports의 엄청난 성공을 확인할 수 있다.
- 정당한 데이터지만 데이터 분석에서 머신러닝의 학습을 뛰어넘는 수치이기에 제거한다.


```python
gs1 = df['Global_Sales'].quantile(0.99)
gs2 = df['Global_Sales'].quantile(0.01)
print(gs1, gs2)
```

    7.167600000000002 0.01



```python
# 하위 1%경우 아웃라이어는 없다고 예측 가능. 상위 1%만 제거해준다.
df= df[df['Global_Sales'] < gs1 ]
```


```python
sns.histplot(data=df, x='Global_Sales')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32d873cc8>




![output_19_1](https://user-images.githubusercontent.com/77723966/112605211-648b8c80-8e5a-11eb-88c1-e3cbcad38d4a.png)




```python
sns.histplot(data=df, x='Critic_Score', bins=16)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32da938c8>




![output_20_1](https://user-images.githubusercontent.com/77723966/112605217-67867d00-8e5a-11eb-83b4-624cba462df9.png)



```python
sns.histplot(data=df['User_Score'].apply(float), bins=16)

# string타입이 있어 float타입으로 모두 바꾸어 표시.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32f04b748>




![output_21_1](https://user-images.githubusercontent.com/77723966/112605228-6b1a0400-8e5a-11eb-99b5-1f5318c14010.png)



- 평론가 보다는 유저들이 점수를 상대적으로 조금 후하게 주는 편이다.
- 두 플롯의 모양이 비슷하고 평균은 각각 80, 8 점대에 머무르며 그 이후 급격히 줄어든다.


```python
sns.histplot(data=df, x='Critic_Count', bins=16)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32f0c0bc8>




![output_23_1](https://user-images.githubusercontent.com/77723966/112605235-6d7c5e00-8e5a-11eb-8882-30be0a674fec.png)




```python
sns.rugplot(data=df, x='User_Count')

# 범위가 방대하고 들쭉날쭉이다. 잘라내기가 필요.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32f358dc8>




![output_24_1](https://user-images.githubusercontent.com/77723966/112605244-710fe500-8e5a-11eb-9beb-b558b6c963e3.png)




```python
uc1 = df['User_Count'].quantile(0.96)
uc1
```




    908.8000000000002




```python
df = df[df['User_Count'] < uc1]
sns.histplot(data=df, x='User_Count')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b32f746dc8>




![output_26_1](https://user-images.githubusercontent.com/77723966/112605271-7836f300-8e5a-11eb-9325-759af520bb64.png)



### 수치형 데이터 관계확인


```python
# 판매액과 상관있을 것 같은 변수들을 jointplot으로 함께 가시화 (평론가 점수와 유저점수와 같은)

sns.jointplot(data=df, x='Critic_Score', y='Global_Sales', kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x2b331176948>




![output_28_1](https://user-images.githubusercontent.com/77723966/112605280-7bca7a00-8e5a-11eb-8ff5-3283dac5ac0b.png)




```python
df['User_Score'] = df['User_Score'].apply(float)
```

    C:\Users\dissi\anaconda31\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



```python
sns.jointplot(data=df, x='User_Score', y='Global_Sales', kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x2b3318d3a08>




![output_30_1](https://user-images.githubusercontent.com/77723966/112605309-81c05b00-8e5a-11eb-99f1-ecb90a80a49c.png)



- 유저점수나 평론가 점수가 판매를 보장하지는 못하지만 상관성은 분명히 있음. (몇가지 아웃라이어가 존재)
- 판매 자체가 저조한것이 디폴트.
- 유저점수 경우 평론가 점수보다 후한 경우가 많아서 상관성이 더 떨어져 보일 수 있다.


```python
sns.jointplot(data=df, x='Critic_Count', y='Global_Sales' ,kind='hist')
```




    <seaborn.axisgrid.JointGrid at 0x2b331a43348>




![output_32_1](https://user-images.githubusercontent.com/77723966/112605319-8553e200-8e5a-11eb-8fa3-74600e357375.png)



- 대부분 20미만으로 평론가가 투입되는데 20미만의 경우 이상보다 저조한 판매게임이 많다.
- 20명 이상의 평론가가 투입되는 데이터의 경우 판매액이 나름 고르게 분포되는것을 확인할 수 있다.
- 20명 이상 평론가가 투입되는 게임의 경우 '기대작'이라고 분류한다면 비교적 높은 판매액 또는 판매액이 고르게 분포된 데이터가 나오는것으로 예측해볼 수 있다.

- 평론가가아닌 User Count 같은 경우 Sales와 큰 상관관계가 있기에 변수로 적합한지 고려해볼 필요가 있다. (많이 판매되었기에 점수 평가에 많이 참여할 수 있기 때문.)


```python
#### 수치형 데이터 간 관계 확인
plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b3335dbc08>




![output_34_1](https://user-images.githubusercontent.com/77723966/112605327-88e76900-8e5a-11eb-94d9-f3ddf8f600fd.png)



- 다른 곳의 sales 액수가 모여 global sales가 되기에 사실 상관성이 높음. 그중에서도 북미판매액이 가장 많은 영향을 미침.
- User_Count도 마찬가지이다. Count가 많을수록 많이 사서 플레이를 했다는 증거이기 때문이다.
- Critic_Score, Critic_Count는 0.3으로 비슷하게 영향을 미친다. 공신력, 기대작으로 이해하면 좋을 듯 하다.
- Critic_Score와 User_Score간 상관관계도 주목할만하다. 평론가의 영향인지 사람의 평가가 비슷한건지는 더욱 분석할 여지가 있다.
- 최근 발매된 게임일수록 점수가 박하게 받는 양상이 있다. (-0.26)

### 범주형 데이터 확인


```python
df.columns
```




    Index(['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales',
           'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score',
           'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'],
          dtype='object')




```python
plt.figure(figsize=(15,5))
sns.boxplot(data=df, x='Platform',y='Global_Sales')
plt.show()
```


![output_38_0](https://user-images.githubusercontent.com/77723966/112605343-8d138680-8e5a-11eb-9076-2497a6917152.png)




```python
plt.figure(figsize=(15,5))
sns.boxplot(data=df, x='Genre',y='Global_Sales')
plt.show()
```


![output_39_0](https://user-images.githubusercontent.com/77723966/112605352-900e7700-8e5a-11eb-8065-c419fbc7e0d9.png)



- 전체적으로 비슷하나 어드벤쳐와 전략물, 롤플레잉 장르의 판매가 저조하다. 스포츠와 특정플랫폼장르의 게임이 판매가 살짝 앞서고 있다.
- Publisher 와 Developer 변수는 숫자가 워낙 많아 가시화 제외.


```python
# 평론가 점수와 유저 점수를 같이 살피기 위해 합쳐준다.

critic_score = df[['Critic_Score']].copy()
critic_score.rename({'Critic_Score' : 'Score'}, axis=1, inplace=True)
critic_score['ScoreBy'] = 'Critics'

user_score = df[['User_Score']].copy() * 10
user_score.rename({'User_Score' : 'Score'}, axis=1, inplace=True)
user_score['ScoreBy'] = 'Users'

scores = pd.concat([critic_score, user_score])
scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Score</th>
      <th>ScoreBy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>96.0</td>
      <td>Critics</td>
    </tr>
    <tr>
      <th>109</th>
      <td>91.0</td>
      <td>Critics</td>
    </tr>
    <tr>
      <th>111</th>
      <td>92.0</td>
      <td>Critics</td>
    </tr>
    <tr>
      <th>113</th>
      <td>82.0</td>
      <td>Critics</td>
    </tr>
    <tr>
      <th>114</th>
      <td>88.0</td>
      <td>Critics</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>16667</th>
      <td>24.0</td>
      <td>Users</td>
    </tr>
    <tr>
      <th>16677</th>
      <td>88.0</td>
      <td>Users</td>
    </tr>
    <tr>
      <th>16696</th>
      <td>76.0</td>
      <td>Users</td>
    </tr>
    <tr>
      <th>16700</th>
      <td>58.0</td>
      <td>Users</td>
    </tr>
    <tr>
      <th>16706</th>
      <td>72.0</td>
      <td>Users</td>
    </tr>
  </tbody>
</table>
<p>12970 rows × 2 columns</p>

</div>



### 범주형 데이터 관계확인


```python
sns.boxplot(data=scores, x='ScoreBy', y='Score')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b332220288>




![output_43_1](https://user-images.githubusercontent.com/77723966/112605374-97358500-8e5a-11eb-8bc7-fa18a3009641.png)




```python
sns. boxplot(data=df, x='Genre', y='Critic_Score')
plt.xticks(rotation=90)
plt.show()
```


![output_44_0](https://user-images.githubusercontent.com/77723966/112605410-9e5c9300-8e5a-11eb-8d8d-02684c2ab1d5.png)




```python
sns. boxplot(data=df, x='Genre', y='User_Score')
plt.xticks(rotation=90)
plt.show()
```


![output_45_0](https://user-images.githubusercontent.com/77723966/112605418-a1578380-8e5a-11eb-9632-f66f1d8addff.png)


- 평론가 그룹은 스포츠에 점수를 후하게 주는편이고 액션, 어드벤쳐 장르에는 박하게 주는 편이다.
- 유저는 평론가 그룹과 다르게 어드벤쳐, 롤플레잉, 플랫폼 장르가 앞서고 있으며 액션과 슈팅게임은 저조하게 주는 양상을 보이고 있다.
- 퍼즐과 전략 장르 게임이 동시에 높다는것은 주목할 만하다. 판매액은 절대적인 수치대비 낮은 수준이지만.


```python
sns.boxplot(data=df, x='Genre', y='Critic_Count')
plt.xticks(rotation=90)
plt.show()
```


![output_47_0](https://user-images.githubusercontent.com/77723966/112605434-a4527400-8e5a-11eb-9170-80db3a3807da.png)




```python
sns.countplot(data=df, x='Genre')
plt.xticks(rotation=90)
plt.show()
```


![output_48_0](https://user-images.githubusercontent.com/77723966/112605444-a74d6480-8e5a-11eb-9d32-a682abe965d3.png)



- 앞서 말했듯 평론가 참여가 많을 수록 사전기대지수가 높다고 예상해볼 수 있는데 슈팅-롤플레잉 우세가 주목할만하다.
- 동시에 판매액이 절대적으로 적은 퍼즐과 전략 장르의 게임(애초에 출시도 상대적으로 적음)의 평론가 참여가 높은데 이 역시 주목할만하다.

## 전처리

### 범주형 데이터 전처리


```python
# Publisher 와 Developer 범주가 많아 끊어주어야 분석에 용이
pb = df['Publisher'].value_counts()
pb
```




    Electronic Arts                902
    Ubisoft                        468
    Activision                     458
    THQ                            301
    Sony Computer Entertainment    287
                                  ... 
    Hudson Entertainment             1
    Phantom EFX                      1
    RTL                              1
    Illusion Softworks               1
    Sunsoft                          1
    Name: Publisher, Length: 253, dtype: int64




```python
plt.plot(range(len(pb)), pb)
```




    [<matplotlib.lines.Line2D at 0x2b33390db88>]




![output_53_1](https://user-images.githubusercontent.com/77723966/112605462-acaaaf00-8e5a-11eb-914c-7a7ebddec5f6.png)




```python
df['Publisher'] = df['Publisher'].apply(lambda s : s if s not in pb[20:] else 'others')

plt.figure(figsize=(15,5))
sns.countplot(data=df, x='Publisher')
plt.xticks(rotation=90)
plt.show()
```

    C:\Users\dissi\anaconda31\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



![output_54_1](https://user-images.githubusercontent.com/77723966/112605473-af0d0900-8e5a-11eb-9a7d-2a25fcb1fe41.png)




```python
plt.figure(figsize=(15,5))
sns.boxplot(data=df, x='Publisher',y='Global_Sales')
plt.xticks(rotation=90)
plt.show()
```

![output_55_0](https://user-images.githubusercontent.com/77723966/112605478-b207f980-8e5a-11eb-9d19-cec9c2566e63.png)



- 닌텐도 압도적. 다음으로 EA(스포츠게임), 워너브라더스, 마이크로스프트 순
- 퍼블리셔 별로 어떤 장르를 출시하는지 분석하는것도 의미있을 듯 보임.


```python
dev = df['Developer'].value_counts()
plt.plot(range(len(dev)), dev)
```




    [<matplotlib.lines.Line2D at 0x2b333f814c8>]




![output_57_1](https://user-images.githubusercontent.com/77723966/112605495-b6341700-8e5a-11eb-91df-f7838a8b8814.png)




```python
df['Developer'] = df['Developer'].apply(lambda s : s if s not in dev[20:] else 'others')

plt.figure(figsize=(15,5))
sns.boxplot(data=df, x='Developer', y='Global_Sales')
plt.xticks(rotation=90)
plt.show()
```

    C:\Users\dissi\anaconda31\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.



![output_58_1](https://user-images.githubusercontent.com/77723966/112605503-b92f0780-8e5a-11eb-9b40-06547366be12.png)



- 닌텐도 유통도 하지만 개발도 한다. 네버소프트, 비쥬얼컨셉, EA 순


```python
df.columns
```




    Index(['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales',
           'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score',
           'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating'],
          dtype='object')




```python
X_cat = df[['Platform', 'Genre', 'Publisher', 'Developer']]
X_cat = pd.get_dummies(X_cat, drop_first=True)
```

### 수치형 데이터 전처리


```python
from sklearn.preprocessing import StandardScaler

X_num = df[['Year_of_Release', 'Critic_Score','Critic_Count']]
scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)
```

- User_Count, User_Score 경우 각각 Sales와 Critic_Score 에 영향을 받았다고 예측할 수 있기에 drop시킴.


```python
X = pd.concat([X_cat, X_scaled], axis=1)
y = df['Global_Sales']
```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 모델 학습 및 평가

#### XGBoost Regression


```python
from xgboost import XGBRegressor

model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=100, n_jobs=4, num_parallel_tree=1,
                 objective='reg:squarederror', random_state=0, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',
                 validate_parameters=1, verbosity=None)



#### Linear Regression


```python
from sklearn.linear_model import LinearRegression

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)



### 평가


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

pred_xgb = model_xgb.predict(X_test)
pred_lr = model_lr.predict(X_test)

print('XGB MAE:', mean_absolute_error(y_test, pred_xgb))
print('XGB RMSE:', sqrt(mean_squared_error(y_test, pred_xgb)))

print('LR MAE:', mean_absolute_error(y_test, pred_lr))
print('LR RMSE:', sqrt(mean_squared_error(y_test, pred_lr)))
```

    XGB MAE: 0.40108377501848275
    XGB RMSE: 0.7042519240022177
    LR MAE: 0.4393590163959862
    LR RMSE: 0.6921270249386282



```python
plt.scatter(y_test, pred_xgb, alpha=0.1)
plt.plot([0, 8], [0,8], '-r')
```




    [<matplotlib.lines.Line2D at 0x2b335b542c8>]




![output_74_1](https://user-images.githubusercontent.com/77723966/112605533-bfbd7f00-8e5a-11eb-8896-e61b3bded638.png)



- 적게 팔린 게임은 overestimate 많이 팔린 게임은 오히려 underestimate 하는 경향이 나타난다.
- 일단 판매액이 일반적으로 적다보니 적은판매액의 데이터에 집중하게되어 높은 판매액 데이터는 오류가 나고 있다.
- 또한 판매액 높은 게임은 아웃라이어이다.


```python
plt.scatter(y_test, pred_lr, alpha=0.1)
plt.plot([0, 8], [0,8], '-r')
```




    [<matplotlib.lines.Line2D at 0x2b333e0be08>]




![output_76_1](https://user-images.githubusercontent.com/77723966/112605542-c2b86f80-8e5a-11eb-9003-9e283f181160.png)



- 가중치 설정으로 음수가 나오는 경우가 생긴다.


```python
plt.figure(figsize=(7, 15))
plt.barh(X.columns, model_xgb.feature_importances_)
plt.show()
```


![output_78_0](https://user-images.githubusercontent.com/77723966/112605551-c5b36000-8e5a-11eb-9c3d-b25059a646b2.png)







