---
layout: post
title:  "Project - 아파트 분양 가격 2013-2020 분석"
description: "공공데이터 분석"
author: SeungRok OH
categories: [Analyze_Project]
---


# 아파트 가격 데이터
- 정부의 대책에도 불구하고 주택에 대한 국민들의 불편함이 계속해서 나오고 있다.
- 부동산 가격이 계속해서 오르고 있다는데 국민들이 가장 선호하는 주택 형태인 아파트에 그 변동이 적용되는지 알아보고자 한다.

# 데이터 셋
- 공공데이터 포털 https://www.data.go.kr/data/15061057/fileData.do
    
## 전국 평균 분양가격(2013년 9월부터 2015년 8월까지)¶
## 주택도시보증공사 전국 신규 민간아파트 분양가격 동향 20201116

# 라이브러리 불러오기 & 폰트설정


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rc("font", family = "Malgun Gothic")
plt.rc("axes", unicode_minus=False)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")
```

# 데이터 불러오기


```python
df_2013 = pd.read_csv("./전국 평균 분양가격(2013년 9월부터 2015년 8월까지).csv", encoding="cp949")
df_2020 = pd.read_csv("./주택도시보증공사_전국 신규 민간아파트 분양가격 동향_20201116.csv", encoding="cp949")
```


```python
df_2013.head()
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
      <th>지역</th>
      <th>2013년12월</th>
      <th>2014년1월</th>
      <th>2014년2월</th>
      <th>2014년3월</th>
      <th>2014년4월</th>
      <th>2014년5월</th>
      <th>2014년6월</th>
      <th>2014년7월</th>
      <th>2014년8월</th>
      <th>...</th>
      <th>2014년11월</th>
      <th>2014년12월</th>
      <th>2015년1월</th>
      <th>2015년2월</th>
      <th>2015년3월</th>
      <th>2015년4월</th>
      <th>2015년5월</th>
      <th>2015년6월</th>
      <th>2015년7월</th>
      <th>2015년8월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>18189</td>
      <td>17925</td>
      <td>17925</td>
      <td>18016</td>
      <td>18098</td>
      <td>19446</td>
      <td>18867</td>
      <td>18742</td>
      <td>19274</td>
      <td>...</td>
      <td>20242</td>
      <td>20269</td>
      <td>20670</td>
      <td>20670</td>
      <td>19415</td>
      <td>18842</td>
      <td>18367</td>
      <td>18374</td>
      <td>18152</td>
      <td>18443</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>8111</td>
      <td>8111</td>
      <td>9078</td>
      <td>8965</td>
      <td>9402</td>
      <td>9501</td>
      <td>9453</td>
      <td>9457</td>
      <td>9411</td>
      <td>...</td>
      <td>9208</td>
      <td>9208</td>
      <td>9204</td>
      <td>9235</td>
      <td>9279</td>
      <td>9327</td>
      <td>9345</td>
      <td>9515</td>
      <td>9559</td>
      <td>9581</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>8080</td>
      <td>8080</td>
      <td>8077</td>
      <td>8101</td>
      <td>8267</td>
      <td>8274</td>
      <td>8360</td>
      <td>8360</td>
      <td>8370</td>
      <td>...</td>
      <td>8439</td>
      <td>8253</td>
      <td>8327</td>
      <td>8416</td>
      <td>8441</td>
      <td>8446</td>
      <td>8568</td>
      <td>8542</td>
      <td>8542</td>
      <td>8795</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>10204</td>
      <td>10204</td>
      <td>10408</td>
      <td>10408</td>
      <td>10000</td>
      <td>9844</td>
      <td>10058</td>
      <td>9974</td>
      <td>9973</td>
      <td>...</td>
      <td>10020</td>
      <td>10020</td>
      <td>10017</td>
      <td>9876</td>
      <td>9876</td>
      <td>9938</td>
      <td>10551</td>
      <td>10443</td>
      <td>10443</td>
      <td>10449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>6098</td>
      <td>7326</td>
      <td>7611</td>
      <td>7346</td>
      <td>7346</td>
      <td>7523</td>
      <td>7659</td>
      <td>7612</td>
      <td>7622</td>
      <td>...</td>
      <td>7752</td>
      <td>7748</td>
      <td>7752</td>
      <td>7756</td>
      <td>7861</td>
      <td>7914</td>
      <td>7877</td>
      <td>7881</td>
      <td>8089</td>
      <td>8231</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df_2020.head()
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격(㎡)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2020.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5185 entries, 0 to 5184
    Data columns (total 5 columns):
     #   Column   Non-Null Count  Dtype 
    ---  ------   --------------  ----- 
     0   지역명      5185 non-null   object
     1   규모구분     5185 non-null   object
     2   연도       5185 non-null   int64 
     3   월        5185 non-null   int64 
     4   분양가격(㎡)  4780 non-null   object
    dtypes: int64(2), object(3)
    memory usage: 202.7+ KB
    

## 필요한 columns 추가 및 결측치 확인
- 2013 ~ 2015까지의 데이터는 가격만 제시되어있고 일반 분양가격(㎡)이 아닌 평당분양가격으로 제시되어 있다.
- 데이터 취합을 위해 가격 기준을 맞추고 새로운 column을 생성한다.


```python
df_2020.isnull().sum()
```




    지역명          0
    규모구분         0
    연도           0
    월            0
    분양가격(㎡)    405
    dtype: int64



- 계산 가능하도록 문자열에서 숫자열로 변경
- 분양가격에서 3.3을 곱한 평당분양가격 column을 추가해 비교가능하도록 설정한다.


```python
df_2020["분양가격"] = pd.to_numeric(df_2020["분양가격(㎡)"], errors='coerce')

df_2020["평당분양가격"] = df_2020["분양가격"] * 3.3
df_2020.head()
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
      <th>지역명</th>
      <th>규모구분</th>
      <th>연도</th>
      <th>월</th>
      <th>분양가격(㎡)</th>
      <th>분양가격</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>전체</td>
      <td>2015</td>
      <td>10</td>
      <td>5841</td>
      <td>5841.0</td>
      <td>19275.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>서울</td>
      <td>전용면적 60㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5652</td>
      <td>5652.0</td>
      <td>18651.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>서울</td>
      <td>전용면적 60㎡초과 85㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5882</td>
      <td>5882.0</td>
      <td>19410.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>서울</td>
      <td>전용면적 85㎡초과 102㎡이하</td>
      <td>2015</td>
      <td>10</td>
      <td>5721</td>
      <td>5721.0</td>
      <td>18879.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>서울</td>
      <td>전용면적 102㎡초과</td>
      <td>2015</td>
      <td>10</td>
      <td>5879</td>
      <td>5879.0</td>
      <td>19400.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2020 = df_2020.drop(["분양가격(㎡)"], axis=1)
```

- 2015이전 데이터는 이후 데이터와 형태가 다르다. 열데이터를 행데이터로 전환하여 데이터의 형태를 맞추어 준다.
- 지역을 지역명으로, 기간을 연도와 월 형태로 나누어준다.


```python
df_2013_melt = df_2013.melt(id_vars="지역",var_name="기간", value_name="평당분양가격")
df_2013_melt.columns =  ["지역명", "기간", "평당분양가격"] 
```


```python
df_2013_melt.head()
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
      <th>지역명</th>
      <th>기간</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2013년12월</td>
      <td>18189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>2013년12월</td>
      <td>8111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>2013년12월</td>
      <td>8080</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>2013년12월</td>
      <td>10204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>2013년12월</td>
      <td>6098</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 연,월 구분하는 함수 만들기

def split_year(date):
    year = date.split("년")[0]
    year = int(year) # 데이터가 int 타입이 되도록
    return year

def split_month(date):
    month = date.split("년")[-1].replace("월","")
    month = int(month)
    return month
```


```python
df_2013_melt["연도"] = df_2013_melt["기간"].apply(split_year)
df_2013_melt["월"] = df_2013_melt["기간"].apply(split_month)
df_2013_melt.sample(5)
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
      <th>지역명</th>
      <th>기간</th>
      <th>평당분양가격</th>
      <th>연도</th>
      <th>월</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>273</th>
      <td>부산</td>
      <td>2015년4월</td>
      <td>9327</td>
      <td>2015</td>
      <td>4</td>
    </tr>
    <tr>
      <th>301</th>
      <td>전북</td>
      <td>2015년5월</td>
      <td>6556</td>
      <td>2015</td>
      <td>5</td>
    </tr>
    <tr>
      <th>250</th>
      <td>전북</td>
      <td>2015년2월</td>
      <td>6583</td>
      <td>2015</td>
      <td>2</td>
    </tr>
    <tr>
      <th>297</th>
      <td>세종</td>
      <td>2015년5월</td>
      <td>8546</td>
      <td>2015</td>
      <td>5</td>
    </tr>
    <tr>
      <th>172</th>
      <td>대구</td>
      <td>2014년10월</td>
      <td>8403</td>
      <td>2014</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# cols 사용할 column명들
cols = ['지역명', '연도', '월', '평당분양가격'] 

# 2015이전 데이터의 경우 규모구분이 따로 설정되어 있지 않아 비교를 위해 2015년이후는 '전체'에 해당하는 데이터만 추출
df_2020_compare = df_2020.loc[df_2020["규모구분"] == "전체",cols].copy()
df_2020_compare.head()
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
      <th>지역명</th>
      <th>연도</th>
      <th>월</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2015</td>
      <td>10</td>
      <td>19275.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>인천</td>
      <td>2015</td>
      <td>10</td>
      <td>10437.9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>경기</td>
      <td>2015</td>
      <td>10</td>
      <td>10355.4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>부산</td>
      <td>2015</td>
      <td>10</td>
      <td>10269.6</td>
    </tr>
    <tr>
      <th>20</th>
      <td>대구</td>
      <td>2015</td>
      <td>10</td>
      <td>8850.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2013_compare = df_2013_melt[cols].copy()
df_2013_compare.head()
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
      <th>지역명</th>
      <th>연도</th>
      <th>월</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2013</td>
      <td>12</td>
      <td>18189</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>2013</td>
      <td>12</td>
      <td>8111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>2013</td>
      <td>12</td>
      <td>8080</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>2013</td>
      <td>12</td>
      <td>10204</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>2013</td>
      <td>12</td>
      <td>6098</td>
    </tr>
  </tbody>
</table>
</div>



## 데이터 결합

- concat을 이용하여 2015년 이전, 이후 데이터를 합쳐준다


```python
df = pd.concat([df_2013_compare, df_2020_compare])
df
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
      <th>지역명</th>
      <th>연도</th>
      <th>월</th>
      <th>평당분양가격</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>서울</td>
      <td>2013</td>
      <td>12</td>
      <td>18189.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>부산</td>
      <td>2013</td>
      <td>12</td>
      <td>8111.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>대구</td>
      <td>2013</td>
      <td>12</td>
      <td>8080.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>인천</td>
      <td>2013</td>
      <td>12</td>
      <td>10204.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>광주</td>
      <td>2013</td>
      <td>12</td>
      <td>6098.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5160</th>
      <td>전북</td>
      <td>2020</td>
      <td>10</td>
      <td>8487.6</td>
    </tr>
    <tr>
      <th>5165</th>
      <td>전남</td>
      <td>2020</td>
      <td>10</td>
      <td>8969.4</td>
    </tr>
    <tr>
      <th>5170</th>
      <td>경북</td>
      <td>2020</td>
      <td>10</td>
      <td>10114.5</td>
    </tr>
    <tr>
      <th>5175</th>
      <td>경남</td>
      <td>2020</td>
      <td>10</td>
      <td>10253.1</td>
    </tr>
    <tr>
      <th>5180</th>
      <td>제주</td>
      <td>2020</td>
      <td>10</td>
      <td>15338.4</td>
    </tr>
  </tbody>
</table>
<p>1394 rows × 4 columns</p>
</div>




```python
# 연도별 data의 개수는?
df["연도"].value_counts(sort=False)
```




    2013     17
    2014    204
    2015    187
    2016    204
    2017    204
    2018    204
    2019    204
    2020    170
    Name: 연도, dtype: int64



# 데이터시각화

## 전국데이터


```python
sns.barplot(data=df, x="연도", y="평당분양가격")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b496ad9b48>




<img width="405" alt="output_26_1" src="https://user-images.githubusercontent.com/77723966/112034375-c1005a80-8b81-11eb-97ac-f7cb152d8a6c.png">



```python
# 히트맵을 사용하여 지역별, 연도별 가격을 확인한다.
table = pd.pivot_table(df, index="연도", columns="지역명", values="평당분양가격").round()

plt.figure(figsize=(12, 7))
sns.heatmap(table, cmap="Blues", annot=True, fmt=".0f")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b4971f8c88>




<img width="679" alt="output_27_1" src="https://user-images.githubusercontent.com/77723966/112034388-c52c7800-8b81-11eb-8c8c-74650437bd4e.png">




```python
# pointplot으로 나타내기

plt.figure(figsize=(12,4))
sns.pointplot(data=df, x="연도", y="평당분양가격", hue="지역명")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
```




    <matplotlib.legend.Legend at 0x2b4973dedc8>




<img width="815" alt="output_28_1" src="https://user-images.githubusercontent.com/77723966/112034400-c8bfff00-8b81-11eb-8763-4c09ca1470c0.png">


## 지역별데이터


```python
plt.figure(figsize=(12,4))
sns.barplot(data=df, x="지역명", y="평당분양가격")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b496a57a48>




<img width="740" alt="output_30_1" src="https://user-images.githubusercontent.com/77723966/112034409-cc538600-8b81-11eb-8571-e4984f2627ec.png">



```python
# violinplot으로 가시화를 더 쉽게하여 나타내본다.
plt.figure(figsize=(20, 4))
sns.violinplot(data=df, x="지역명", y="평당분양가격")
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b49794de88>




<img width="1187" alt="output_31_1" src="https://user-images.githubusercontent.com/77723966/112034422-d07fa380-8b81-11eb-9e86-6f3872247542.png">



```python
# swarmplot으로 연도별 지역의 평균가격을 확인해본다.
plt.figure(figsize=(12, 4))
sns.swarmplot(data=df, x="지역명", y="평당분양가격", hue="연도")
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
```




    <matplotlib.legend.Legend at 0x2b498bc7048>




<img width="818" alt="output_32_1" src="https://user-images.githubusercontent.com/77723966/112034447-d5dcee00-8b81-11eb-8ae2-bbb8f8c1b055.png">



- 5.1 전국데이터 확인결과 해가 지나갈 수록 아파트 분양가격이 상승하고 있음을 확인할 수 있다.
- 5.1 heatmap에서 확인할 수 있듯, 수도권(서울, 경기)의 집값이 다른 지역보다 높음을 알 수 있다.

- 5.1 pointplot과 5.2지역별데이터의 경우 서울과 다른 지역의 분양가격 차이를 더욱 확연히 알 수 있다.
- 5.2 swarmplot에서 확인할 수 있듯, 서울의 경우 2018년 이후의 데이터가 뚜렷하게 상승하고 있음을 알 수 있다. 
- 전국의 데이터 역시 2018년 이후상승하고 있음을 알 수 있다.(갈색, 연보라색, 회색) 

# 2015년 이후 아파트 가격 데이터 자세하게 살펴보기

- 2015년 이후 데이터는 이전과 다르게 교모구분 column을 통해 규모구분 별 데이터를 더욱 자세히 다룰 수 있다.


```python
# 이런식으로 확인 가능.
df_2020.groupby(["지역명","규모구분"])["평당분양가격"].mean().round()
```




    지역명  규모구분             
    강원   전용면적 102㎡초과          9149.0
         전용면적 60㎡이하           7865.0
         전용면적 60㎡초과 85㎡이하     7752.0
         전용면적 85㎡초과 102㎡이하    8868.0
         전체                   7752.0
                               ...  
    충북   전용면적 102㎡초과          8341.0
         전용면적 60㎡이하           7240.0
         전용면적 60㎡초과 85㎡이하     7365.0
         전용면적 85㎡초과 102㎡이하    8467.0
         전체                   7331.0
    Name: 평당분양가격, Length: 85, dtype: float64




```python
# 지역과 상관없는 규모구분별 가격
df_2020.groupby(["규모구분"])["평당분양가격"].mean()
```




    규모구분
    전용면적 102㎡초과          11882.902845
    전용면적 60㎡이하           10700.634247
    전용면적 60㎡초과 85㎡이하     10591.392729
    전용면적 85㎡초과 102㎡이하    11620.084559
    전체                   10594.894627
    Name: 평당분양가격, dtype: float64




```python
df_2020.groupby(["규모구분"])["평당분양가격"].mean().plot.bar(rot=30)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2b498a2c148>



<img width="389" alt="output_37_1" src="https://user-images.githubusercontent.com/77723966/112034478-dc6b6580-8b81-11eb-974f-31bc3b536767.png">



- 60㎡이하의 아파트의 평균가격이 60㎡초과 85㎡이하의 아파트의 평균가격보다 높음을 확인할 수 있다.
- 주어진 데이터로 유추하였을때 지역에 비해 수도권의 아파트 형태가 더욱 다양하고, 좁은 면적의 아파트 역시 많을것으로 가정한다면
- 수도권의 적은 평수 아파트가 평균가격 인상에 영향을 미쳤음을 유추할 수 있다.


```python

```
