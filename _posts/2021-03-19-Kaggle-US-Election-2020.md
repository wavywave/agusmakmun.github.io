---
layout: post
title:  "Kaggle - US Election 2020"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---


# US Election 2020

### 데이터 셋
https://www.kaggle.com/unanimad/us-election-2020 (선거관련)
https://www.kaggle.com/muonneutrino/us-census-demographic-data (인구관련)

- 대통령 선거 이전에 보여지는 모습을 통해 인종, 상황, 배경이라는 변수에 따라 투표의 양상이 어떻게 벌어지는지 확인할 수 있는 데이터 셋이다.

## 라이브러리 설정 및 데이터 읽어들이기

- president_country_candidate.csv : 대통령 투표 결과
- governors_country_candidate.csv : 카운티 지사 투표 결과
- senate_country_candidate.csv : 상원의원 투표 결과
- house_candidate.csv : 하원의원 투표 결과


- state : 주
- county : 카운티(군)
- district : 지구
- candidate : 후보자
- party : 후보자의 소속 정당
- total_votes: 득표 수
- won : 지역 투표 우승 여부


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_pres = pd.read_csv("us-election-2020/president_county_candidate.csv")
df_gov = pd.read_csv("us-election-2020/governors_county_candidate.csv")
# 우선 두가지 사용.

# df_sen = pd.read_csv('us-election-2020/senate_country_candidate.csv')
# df_hou = pd.read_csv('us-election-2020/house_candidate.csv')

df_census = pd.read_csv("./us-census-demographic/acs2017_county_data.csv") # 인구 관련 조사
```


```python
# State Code 관련 부가 자료
state_code = pd.read_html('https://www.infoplease.com/us/postal-information/state-abbreviations-and-state-postal-codes')[0]
```

## EDA 및 기초 통계 분석


```python
df_pres.head()
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
      <th>state</th>
      <th>county</th>
      <th>candidate</th>
      <th>party</th>
      <th>total_votes</th>
      <th>won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Joe Biden</td>
      <td>DEM</td>
      <td>44552</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Donald Trump</td>
      <td>REP</td>
      <td>41009</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Jo Jorgensen</td>
      <td>LIB</td>
      <td>1044</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Howie Hawkins</td>
      <td>GRN</td>
      <td>420</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Delaware</td>
      <td>New Castle County</td>
      <td>Joe Biden</td>
      <td>DEM</td>
      <td>195034</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_pres['candidate'].unique()
```




    array(['Joe Biden', 'Donald Trump', 'Jo Jorgensen', 'Howie Hawkins',
           ' Write-ins', 'Gloria La Riva', 'Brock Pierce',
           'Rocky De La Fuente', 'Don Blankenship', 'Kanye West',
           'Brian Carroll', 'Ricki Sue King', 'Jade Simmons',
           'President Boddie', 'Bill Hammons', 'Tom Hoefling',
           'Alyson Kennedy', 'Jerome Segal', 'Phil Collins',
           ' None of these candidates', 'Sheila Samm Tittle', 'Dario Hunter',
           'Joe McHugh', 'Christopher LaFontaine', 'Keith McCormic',
           'Brooke Paige', 'Gary Swing', 'Richard Duncan', 'Blake Huber',
           'Kyle Kopitke', 'Zachary Scalf', 'Jesse Ventura', 'Connie Gammon',
           'John Richard Myers', 'Mark Charles', 'Princess Jacob-Fambro',
           'Joseph Kishore', 'Jordan Scott'], dtype=object)




```python
df_pres.loc[df_pres['candidate'] == 'Jo Jorgensen']['total_votes'].sum()
```




    1874183




```python
df_gov.head()
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
      <th>state</th>
      <th>county</th>
      <th>candidate</th>
      <th>party</th>
      <th>votes</th>
      <th>won</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>John Carney</td>
      <td>DEM</td>
      <td>44352</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Julianne Murray</td>
      <td>REP</td>
      <td>39332</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>Kathy DeMatteis</td>
      <td>IPD</td>
      <td>1115</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Delaware</td>
      <td>Kent County</td>
      <td>John Machurek</td>
      <td>LIB</td>
      <td>616</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Delaware</td>
      <td>New Castle County</td>
      <td>John Carney</td>
      <td>DEM</td>
      <td>191678</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.set_option('display.max_columns', None)
```


```python
df_census.head()
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
      <th>CountyId</th>
      <th>State</th>
      <th>County</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>VotingAgeCitizen</th>
      <th>Income</th>
      <th>IncomeErr</th>
      <th>IncomePerCap</th>
      <th>IncomePerCapErr</th>
      <th>Poverty</th>
      <th>ChildPoverty</th>
      <th>Professional</th>
      <th>Service</th>
      <th>Office</th>
      <th>Construction</th>
      <th>Production</th>
      <th>Drive</th>
      <th>Carpool</th>
      <th>Transit</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga County</td>
      <td>55036</td>
      <td>26899</td>
      <td>28137</td>
      <td>2.7</td>
      <td>75.4</td>
      <td>18.9</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>41016</td>
      <td>55317</td>
      <td>2838</td>
      <td>27824</td>
      <td>2024</td>
      <td>13.7</td>
      <td>20.1</td>
      <td>35.3</td>
      <td>18.0</td>
      <td>23.2</td>
      <td>8.1</td>
      <td>15.4</td>
      <td>86.0</td>
      <td>9.6</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>25.8</td>
      <td>24112</td>
      <td>74.1</td>
      <td>20.2</td>
      <td>5.6</td>
      <td>0.1</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin County</td>
      <td>203360</td>
      <td>99527</td>
      <td>103833</td>
      <td>4.4</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>155376</td>
      <td>52562</td>
      <td>1348</td>
      <td>29364</td>
      <td>735</td>
      <td>11.8</td>
      <td>16.1</td>
      <td>35.7</td>
      <td>18.2</td>
      <td>25.6</td>
      <td>9.7</td>
      <td>10.8</td>
      <td>84.7</td>
      <td>7.6</td>
      <td>0.1</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>5.6</td>
      <td>27.0</td>
      <td>89527</td>
      <td>80.7</td>
      <td>12.9</td>
      <td>6.3</td>
      <td>0.1</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour County</td>
      <td>26201</td>
      <td>13976</td>
      <td>12225</td>
      <td>4.2</td>
      <td>45.7</td>
      <td>47.8</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>20269</td>
      <td>33368</td>
      <td>2551</td>
      <td>17561</td>
      <td>798</td>
      <td>27.2</td>
      <td>44.9</td>
      <td>25.0</td>
      <td>16.8</td>
      <td>22.6</td>
      <td>11.5</td>
      <td>24.1</td>
      <td>83.4</td>
      <td>11.1</td>
      <td>0.3</td>
      <td>2.2</td>
      <td>1.7</td>
      <td>1.3</td>
      <td>23.4</td>
      <td>8878</td>
      <td>74.1</td>
      <td>19.1</td>
      <td>6.5</td>
      <td>0.3</td>
      <td>12.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb County</td>
      <td>22580</td>
      <td>12251</td>
      <td>10329</td>
      <td>2.4</td>
      <td>74.6</td>
      <td>22.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17662</td>
      <td>43404</td>
      <td>3431</td>
      <td>20911</td>
      <td>1889</td>
      <td>15.2</td>
      <td>26.6</td>
      <td>24.4</td>
      <td>17.6</td>
      <td>19.7</td>
      <td>15.9</td>
      <td>22.4</td>
      <td>86.4</td>
      <td>9.5</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>1.5</td>
      <td>30.0</td>
      <td>8171</td>
      <td>76.0</td>
      <td>17.4</td>
      <td>6.3</td>
      <td>0.3</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>Alabama</td>
      <td>Blount County</td>
      <td>57667</td>
      <td>28490</td>
      <td>29177</td>
      <td>9.0</td>
      <td>87.4</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>42513</td>
      <td>47412</td>
      <td>2630</td>
      <td>22021</td>
      <td>850</td>
      <td>15.6</td>
      <td>25.4</td>
      <td>28.5</td>
      <td>12.9</td>
      <td>23.3</td>
      <td>15.8</td>
      <td>19.5</td>
      <td>86.8</td>
      <td>10.2</td>
      <td>0.1</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>2.1</td>
      <td>35.0</td>
      <td>21380</td>
      <td>83.9</td>
      <td>11.9</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>



- 숫자로 표기되어있는 data, 백분율로 표기되어 있는 data가 있음. 백분율 경우 TotalPop을 가중치로 두어 전처리 해야 함.


```python
state_code.head()
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
      <th>State/District</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Ala.</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>Ariz.</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>Ark.</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>Calif.</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>



### County별 통계로 데이터프레임 구조 변경


```python
data = df_pres.loc[df_pres['party'].apply(lambda s: str(s) in ['DEM', 'REP'])]

table_pres = pd.pivot_table(data=data, index=['state', 'county'], columns='party', values='total_votes')
table_pres.rename({'DEM':'Pres_DEM', 'REP':'Pres_REP'}, axis=1, inplace=True)
table_pres
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
      <th>party</th>
      <th>Pres_DEM</th>
      <th>Pres_REP</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Alabama</th>
      <th>Autauga County</th>
      <td>7503</td>
      <td>19838</td>
    </tr>
    <tr>
      <th>Baldwin County</th>
      <td>24578</td>
      <td>83544</td>
    </tr>
    <tr>
      <th>Barbour County</th>
      <td>4816</td>
      <td>5622</td>
    </tr>
    <tr>
      <th>Bibb County</th>
      <td>1986</td>
      <td>7525</td>
    </tr>
    <tr>
      <th>Blount County</th>
      <td>2640</td>
      <td>24711</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Wyoming</th>
      <th>Sweetwater County</th>
      <td>3823</td>
      <td>12229</td>
    </tr>
    <tr>
      <th>Teton County</th>
      <td>9848</td>
      <td>4341</td>
    </tr>
    <tr>
      <th>Uinta County</th>
      <td>1591</td>
      <td>7496</td>
    </tr>
    <tr>
      <th>Washakie County</th>
      <td>651</td>
      <td>3245</td>
    </tr>
    <tr>
      <th>Weston County</th>
      <td>360</td>
      <td>3107</td>
    </tr>
  </tbody>
</table>
<p>4633 rows × 2 columns</p>
</div>




```python
table_pres.isna().sum()
```




    party
    Pres_DEM    0
    Pres_REP    0
    dtype: int64




```python
data2 = df_gov.loc[df_gov['party'].apply(lambda s: str(s) in ['DEM', 'REP'])]

table_gov = pd.pivot_table(data=data2, index=['state', 'county'], columns='party', values='votes')
table_gov.rename({'DEM':'Gov_DEM', 'REP':'Gov_REP'}, axis=1, inplace=True)
table_gov
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
      <th>party</th>
      <th>Gov_DEM</th>
      <th>Gov_REP</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Delaware</th>
      <th>Kent County</th>
      <td>44352</td>
      <td>39332</td>
    </tr>
    <tr>
      <th>New Castle County</th>
      <td>191678</td>
      <td>82545</td>
    </tr>
    <tr>
      <th>Sussex County</th>
      <td>56873</td>
      <td>68435</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Indiana</th>
      <th>Adams County</th>
      <td>2143</td>
      <td>9441</td>
    </tr>
    <tr>
      <th>Allen County</th>
      <td>53895</td>
      <td>98406</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">West Virginia</th>
      <th>Webster County</th>
      <td>659</td>
      <td>2552</td>
    </tr>
    <tr>
      <th>Wetzel County</th>
      <td>1727</td>
      <td>4559</td>
    </tr>
    <tr>
      <th>Wirt County</th>
      <td>483</td>
      <td>1947</td>
    </tr>
    <tr>
      <th>Wood County</th>
      <td>9933</td>
      <td>26232</td>
    </tr>
    <tr>
      <th>Wyoming County</th>
      <td>1240</td>
      <td>6941</td>
    </tr>
  </tbody>
</table>
<p>1025 rows × 2 columns</p>
</div>




```python
table_gov.isna().sum()
```




    party
    Gov_DEM    0
    Gov_REP    0
    dtype: int64




```python
df_census.columns
```




    Index(['CountyId', 'State', 'County', 'TotalPop', 'Men', 'Women', 'Hispanic',
           'White', 'Black', 'Native', 'Asian', 'Pacific', 'VotingAgeCitizen',
           'Income', 'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'Poverty',
           'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
           'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
           'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
           'SelfEmployed', 'FamilyWork', 'Unemployment'],
          dtype='object')




```python
df_census.drop(['Income', 'IncomeErr', 'IncomePerCapErr'], axis=1, inplace=True)
```


```python
# state, county 컬럼 소문자로
df_census.rename({'State':'state', 'County':'county'},axis=1, inplace=True)
df_census.drop('CountyId', axis=1, inplace=True)
df_census.set_index(['state', 'county'], inplace=True)
df_census
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
      <th></th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Women</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>VotingAgeCitizen</th>
      <th>IncomePerCap</th>
      <th>Poverty</th>
      <th>ChildPoverty</th>
      <th>Professional</th>
      <th>Service</th>
      <th>Office</th>
      <th>Construction</th>
      <th>Production</th>
      <th>Drive</th>
      <th>Carpool</th>
      <th>Transit</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Alabama</th>
      <th>Autauga County</th>
      <td>55036</td>
      <td>26899</td>
      <td>28137</td>
      <td>2.7</td>
      <td>75.4</td>
      <td>18.9</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>41016</td>
      <td>27824</td>
      <td>13.7</td>
      <td>20.1</td>
      <td>35.3</td>
      <td>18.0</td>
      <td>23.2</td>
      <td>8.1</td>
      <td>15.4</td>
      <td>86.0</td>
      <td>9.6</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>25.8</td>
      <td>24112</td>
      <td>74.1</td>
      <td>20.2</td>
      <td>5.6</td>
      <td>0.1</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>Baldwin County</th>
      <td>203360</td>
      <td>99527</td>
      <td>103833</td>
      <td>4.4</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>155376</td>
      <td>29364</td>
      <td>11.8</td>
      <td>16.1</td>
      <td>35.7</td>
      <td>18.2</td>
      <td>25.6</td>
      <td>9.7</td>
      <td>10.8</td>
      <td>84.7</td>
      <td>7.6</td>
      <td>0.1</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>5.6</td>
      <td>27.0</td>
      <td>89527</td>
      <td>80.7</td>
      <td>12.9</td>
      <td>6.3</td>
      <td>0.1</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>Barbour County</th>
      <td>26201</td>
      <td>13976</td>
      <td>12225</td>
      <td>4.2</td>
      <td>45.7</td>
      <td>47.8</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>20269</td>
      <td>17561</td>
      <td>27.2</td>
      <td>44.9</td>
      <td>25.0</td>
      <td>16.8</td>
      <td>22.6</td>
      <td>11.5</td>
      <td>24.1</td>
      <td>83.4</td>
      <td>11.1</td>
      <td>0.3</td>
      <td>2.2</td>
      <td>1.7</td>
      <td>1.3</td>
      <td>23.4</td>
      <td>8878</td>
      <td>74.1</td>
      <td>19.1</td>
      <td>6.5</td>
      <td>0.3</td>
      <td>12.4</td>
    </tr>
    <tr>
      <th>Bibb County</th>
      <td>22580</td>
      <td>12251</td>
      <td>10329</td>
      <td>2.4</td>
      <td>74.6</td>
      <td>22.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17662</td>
      <td>20911</td>
      <td>15.2</td>
      <td>26.6</td>
      <td>24.4</td>
      <td>17.6</td>
      <td>19.7</td>
      <td>15.9</td>
      <td>22.4</td>
      <td>86.4</td>
      <td>9.5</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>1.5</td>
      <td>30.0</td>
      <td>8171</td>
      <td>76.0</td>
      <td>17.4</td>
      <td>6.3</td>
      <td>0.3</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>Blount County</th>
      <td>57667</td>
      <td>28490</td>
      <td>29177</td>
      <td>9.0</td>
      <td>87.4</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>42513</td>
      <td>22021</td>
      <td>15.6</td>
      <td>25.4</td>
      <td>28.5</td>
      <td>12.9</td>
      <td>23.3</td>
      <td>15.8</td>
      <td>19.5</td>
      <td>86.8</td>
      <td>10.2</td>
      <td>0.1</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>2.1</td>
      <td>35.0</td>
      <td>21380</td>
      <td>83.9</td>
      <td>11.9</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Puerto Rico</th>
      <th>Vega Baja Municipio</th>
      <td>54754</td>
      <td>26269</td>
      <td>28485</td>
      <td>96.7</td>
      <td>3.1</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>42838</td>
      <td>10197</td>
      <td>43.8</td>
      <td>49.4</td>
      <td>28.6</td>
      <td>20.2</td>
      <td>25.9</td>
      <td>11.1</td>
      <td>14.2</td>
      <td>92.0</td>
      <td>4.2</td>
      <td>0.9</td>
      <td>1.4</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>31.6</td>
      <td>14234</td>
      <td>76.2</td>
      <td>19.3</td>
      <td>4.3</td>
      <td>0.2</td>
      <td>16.8</td>
    </tr>
    <tr>
      <th>Vieques Municipio</th>
      <td>8931</td>
      <td>4351</td>
      <td>4580</td>
      <td>95.7</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7045</td>
      <td>11136</td>
      <td>36.8</td>
      <td>68.2</td>
      <td>20.9</td>
      <td>38.4</td>
      <td>16.4</td>
      <td>16.9</td>
      <td>7.3</td>
      <td>76.3</td>
      <td>16.9</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.7</td>
      <td>14.9</td>
      <td>2927</td>
      <td>40.7</td>
      <td>40.9</td>
      <td>18.4</td>
      <td>0.0</td>
      <td>12.8</td>
    </tr>
    <tr>
      <th>Villalba Municipio</th>
      <td>23659</td>
      <td>11510</td>
      <td>12149</td>
      <td>99.7</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>18053</td>
      <td>10449</td>
      <td>50.0</td>
      <td>67.9</td>
      <td>22.5</td>
      <td>21.2</td>
      <td>22.7</td>
      <td>14.1</td>
      <td>19.5</td>
      <td>83.1</td>
      <td>11.8</td>
      <td>0.1</td>
      <td>2.1</td>
      <td>0.0</td>
      <td>2.8</td>
      <td>28.4</td>
      <td>6873</td>
      <td>59.2</td>
      <td>30.2</td>
      <td>10.4</td>
      <td>0.2</td>
      <td>24.8</td>
    </tr>
    <tr>
      <th>Yabucoa Municipio</th>
      <td>35025</td>
      <td>16984</td>
      <td>18041</td>
      <td>99.9</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>27523</td>
      <td>8672</td>
      <td>52.4</td>
      <td>62.1</td>
      <td>27.7</td>
      <td>26.0</td>
      <td>20.7</td>
      <td>9.5</td>
      <td>16.0</td>
      <td>87.6</td>
      <td>9.2</td>
      <td>0.0</td>
      <td>1.4</td>
      <td>1.8</td>
      <td>0.1</td>
      <td>30.5</td>
      <td>7878</td>
      <td>62.7</td>
      <td>30.9</td>
      <td>6.3</td>
      <td>0.0</td>
      <td>25.4</td>
    </tr>
    <tr>
      <th>Yauco Municipio</th>
      <td>37585</td>
      <td>18052</td>
      <td>19533</td>
      <td>99.8</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>29763</td>
      <td>8124</td>
      <td>50.4</td>
      <td>58.2</td>
      <td>30.4</td>
      <td>20.2</td>
      <td>25.6</td>
      <td>12.6</td>
      <td>11.3</td>
      <td>82.8</td>
      <td>8.2</td>
      <td>2.2</td>
      <td>1.7</td>
      <td>0.1</td>
      <td>5.0</td>
      <td>24.4</td>
      <td>8995</td>
      <td>66.4</td>
      <td>28.7</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
<p>3220 rows × 31 columns</p>
</div>




```python
# 다중공선성을 피하기 위해 총인구컬럼과 겹치게 되는 남성-여성인구수, 유권자수, 고용인수를 비율로 바꿔줌.
df_census.drop('Women', axis=1, inplace=True) # 남성아니면 여성이므로 다중공선성 제거를 위해 여성컬럼 제거.

df_census['Men'] /= df_census['TotalPop']
```


```python
df_census['VotingAgeCitizen'] /= df_census['TotalPop']
df_census['Employed'] /= df_census['TotalPop']
```


```python
df_census.head()
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
      <th></th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>VotingAgeCitizen</th>
      <th>IncomePerCap</th>
      <th>Poverty</th>
      <th>ChildPoverty</th>
      <th>Professional</th>
      <th>Service</th>
      <th>Office</th>
      <th>Construction</th>
      <th>Production</th>
      <th>Drive</th>
      <th>Carpool</th>
      <th>Transit</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Alabama</th>
      <th>Autauga County</th>
      <td>55036</td>
      <td>0.488753</td>
      <td>2.7</td>
      <td>75.4</td>
      <td>18.9</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0.745258</td>
      <td>27824</td>
      <td>13.7</td>
      <td>20.1</td>
      <td>35.3</td>
      <td>18.0</td>
      <td>23.2</td>
      <td>8.1</td>
      <td>15.4</td>
      <td>86.0</td>
      <td>9.6</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>25.8</td>
      <td>0.438113</td>
      <td>74.1</td>
      <td>20.2</td>
      <td>5.6</td>
      <td>0.1</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>Baldwin County</th>
      <td>203360</td>
      <td>0.489413</td>
      <td>4.4</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>0.764044</td>
      <td>29364</td>
      <td>11.8</td>
      <td>16.1</td>
      <td>35.7</td>
      <td>18.2</td>
      <td>25.6</td>
      <td>9.7</td>
      <td>10.8</td>
      <td>84.7</td>
      <td>7.6</td>
      <td>0.1</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>5.6</td>
      <td>27.0</td>
      <td>0.440239</td>
      <td>80.7</td>
      <td>12.9</td>
      <td>6.3</td>
      <td>0.1</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>Barbour County</th>
      <td>26201</td>
      <td>0.533415</td>
      <td>4.2</td>
      <td>45.7</td>
      <td>47.8</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.773596</td>
      <td>17561</td>
      <td>27.2</td>
      <td>44.9</td>
      <td>25.0</td>
      <td>16.8</td>
      <td>22.6</td>
      <td>11.5</td>
      <td>24.1</td>
      <td>83.4</td>
      <td>11.1</td>
      <td>0.3</td>
      <td>2.2</td>
      <td>1.7</td>
      <td>1.3</td>
      <td>23.4</td>
      <td>0.338842</td>
      <td>74.1</td>
      <td>19.1</td>
      <td>6.5</td>
      <td>0.3</td>
      <td>12.4</td>
    </tr>
    <tr>
      <th>Bibb County</th>
      <td>22580</td>
      <td>0.542560</td>
      <td>2.4</td>
      <td>74.6</td>
      <td>22.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.782197</td>
      <td>20911</td>
      <td>15.2</td>
      <td>26.6</td>
      <td>24.4</td>
      <td>17.6</td>
      <td>19.7</td>
      <td>15.9</td>
      <td>22.4</td>
      <td>86.4</td>
      <td>9.5</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>1.5</td>
      <td>30.0</td>
      <td>0.361869</td>
      <td>76.0</td>
      <td>17.4</td>
      <td>6.3</td>
      <td>0.3</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>Blount County</th>
      <td>57667</td>
      <td>0.494043</td>
      <td>9.0</td>
      <td>87.4</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.737215</td>
      <td>22021</td>
      <td>15.6</td>
      <td>25.4</td>
      <td>28.5</td>
      <td>12.9</td>
      <td>23.3</td>
      <td>15.8</td>
      <td>19.5</td>
      <td>86.8</td>
      <td>10.2</td>
      <td>0.1</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>2.1</td>
      <td>35.0</td>
      <td>0.370749</td>
      <td>83.9</td>
      <td>11.9</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 세 가지 데이터프레임 통합.
df = pd.concat([table_pres, table_gov, df_census], axis=1)
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
      <th></th>
      <th>Pres_DEM</th>
      <th>Pres_REP</th>
      <th>Gov_DEM</th>
      <th>Gov_REP</th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>VotingAgeCitizen</th>
      <th>IncomePerCap</th>
      <th>Poverty</th>
      <th>ChildPoverty</th>
      <th>Professional</th>
      <th>Service</th>
      <th>Office</th>
      <th>Construction</th>
      <th>Production</th>
      <th>Drive</th>
      <th>Carpool</th>
      <th>Transit</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">Alabama</th>
      <th>Autauga County</th>
      <td>7503.0</td>
      <td>19838.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>55036.0</td>
      <td>0.488753</td>
      <td>2.7</td>
      <td>75.4</td>
      <td>18.9</td>
      <td>0.3</td>
      <td>0.9</td>
      <td>0.0</td>
      <td>0.745258</td>
      <td>27824.0</td>
      <td>13.7</td>
      <td>20.1</td>
      <td>35.3</td>
      <td>18.0</td>
      <td>23.2</td>
      <td>8.1</td>
      <td>15.4</td>
      <td>86.0</td>
      <td>9.6</td>
      <td>0.1</td>
      <td>0.6</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>25.8</td>
      <td>0.438113</td>
      <td>74.1</td>
      <td>20.2</td>
      <td>5.6</td>
      <td>0.1</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>Baldwin County</th>
      <td>24578.0</td>
      <td>83544.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>203360.0</td>
      <td>0.489413</td>
      <td>4.4</td>
      <td>83.1</td>
      <td>9.5</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>0.764044</td>
      <td>29364.0</td>
      <td>11.8</td>
      <td>16.1</td>
      <td>35.7</td>
      <td>18.2</td>
      <td>25.6</td>
      <td>9.7</td>
      <td>10.8</td>
      <td>84.7</td>
      <td>7.6</td>
      <td>0.1</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>5.6</td>
      <td>27.0</td>
      <td>0.440239</td>
      <td>80.7</td>
      <td>12.9</td>
      <td>6.3</td>
      <td>0.1</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>Barbour County</th>
      <td>4816.0</td>
      <td>5622.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>26201.0</td>
      <td>0.533415</td>
      <td>4.2</td>
      <td>45.7</td>
      <td>47.8</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.773596</td>
      <td>17561.0</td>
      <td>27.2</td>
      <td>44.9</td>
      <td>25.0</td>
      <td>16.8</td>
      <td>22.6</td>
      <td>11.5</td>
      <td>24.1</td>
      <td>83.4</td>
      <td>11.1</td>
      <td>0.3</td>
      <td>2.2</td>
      <td>1.7</td>
      <td>1.3</td>
      <td>23.4</td>
      <td>0.338842</td>
      <td>74.1</td>
      <td>19.1</td>
      <td>6.5</td>
      <td>0.3</td>
      <td>12.4</td>
    </tr>
    <tr>
      <th>Bibb County</th>
      <td>1986.0</td>
      <td>7525.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22580.0</td>
      <td>0.542560</td>
      <td>2.4</td>
      <td>74.6</td>
      <td>22.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.782197</td>
      <td>20911.0</td>
      <td>15.2</td>
      <td>26.6</td>
      <td>24.4</td>
      <td>17.6</td>
      <td>19.7</td>
      <td>15.9</td>
      <td>22.4</td>
      <td>86.4</td>
      <td>9.5</td>
      <td>0.7</td>
      <td>0.3</td>
      <td>1.7</td>
      <td>1.5</td>
      <td>30.0</td>
      <td>0.361869</td>
      <td>76.0</td>
      <td>17.4</td>
      <td>6.3</td>
      <td>0.3</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>Blount County</th>
      <td>2640.0</td>
      <td>24711.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>57667.0</td>
      <td>0.494043</td>
      <td>9.0</td>
      <td>87.4</td>
      <td>1.5</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.737215</td>
      <td>22021.0</td>
      <td>15.6</td>
      <td>25.4</td>
      <td>28.5</td>
      <td>12.9</td>
      <td>23.3</td>
      <td>15.8</td>
      <td>19.5</td>
      <td>86.8</td>
      <td>10.2</td>
      <td>0.1</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>2.1</td>
      <td>35.0</td>
      <td>0.370749</td>
      <td>83.9</td>
      <td>11.9</td>
      <td>4.0</td>
      <td>0.1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">Wyoming</th>
      <th>Sweetwater County</th>
      <td>3823.0</td>
      <td>12229.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44527.0</td>
      <td>0.516114</td>
      <td>16.0</td>
      <td>79.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>0.696768</td>
      <td>31700.0</td>
      <td>12.0</td>
      <td>15.7</td>
      <td>27.7</td>
      <td>16.1</td>
      <td>20.0</td>
      <td>20.8</td>
      <td>15.4</td>
      <td>77.5</td>
      <td>14.4</td>
      <td>2.6</td>
      <td>2.8</td>
      <td>1.3</td>
      <td>1.5</td>
      <td>20.5</td>
      <td>0.510679</td>
      <td>78.4</td>
      <td>17.8</td>
      <td>3.8</td>
      <td>0.0</td>
      <td>5.2</td>
    </tr>
    <tr>
      <th>Teton County</th>
      <td>9848.0</td>
      <td>4341.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>22923.0</td>
      <td>0.530864</td>
      <td>15.0</td>
      <td>81.5</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>2.2</td>
      <td>0.0</td>
      <td>0.728177</td>
      <td>49200.0</td>
      <td>6.8</td>
      <td>2.8</td>
      <td>39.4</td>
      <td>25.4</td>
      <td>17.0</td>
      <td>11.7</td>
      <td>6.5</td>
      <td>68.3</td>
      <td>6.7</td>
      <td>3.8</td>
      <td>11.7</td>
      <td>3.8</td>
      <td>5.7</td>
      <td>14.3</td>
      <td>0.632203</td>
      <td>82.1</td>
      <td>11.4</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>Uinta County</th>
      <td>1591.0</td>
      <td>7496.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20758.0</td>
      <td>0.510309</td>
      <td>9.1</td>
      <td>87.7</td>
      <td>0.1</td>
      <td>0.9</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.685760</td>
      <td>27115.0</td>
      <td>14.9</td>
      <td>20.0</td>
      <td>30.4</td>
      <td>19.4</td>
      <td>18.1</td>
      <td>16.1</td>
      <td>16.1</td>
      <td>77.4</td>
      <td>14.9</td>
      <td>3.3</td>
      <td>1.1</td>
      <td>1.3</td>
      <td>2.0</td>
      <td>19.9</td>
      <td>0.459004</td>
      <td>71.5</td>
      <td>21.5</td>
      <td>6.6</td>
      <td>0.4</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>Washakie County</th>
      <td>651.0</td>
      <td>3245.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8253.0</td>
      <td>0.498970</td>
      <td>14.2</td>
      <td>82.2</td>
      <td>0.3</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>0.0</td>
      <td>0.742154</td>
      <td>27345.0</td>
      <td>12.8</td>
      <td>17.5</td>
      <td>32.1</td>
      <td>16.3</td>
      <td>17.6</td>
      <td>18.8</td>
      <td>15.3</td>
      <td>77.2</td>
      <td>10.2</td>
      <td>0.0</td>
      <td>6.9</td>
      <td>1.3</td>
      <td>4.4</td>
      <td>14.3</td>
      <td>0.464437</td>
      <td>69.8</td>
      <td>22.0</td>
      <td>8.1</td>
      <td>0.2</td>
      <td>6.1</td>
    </tr>
    <tr>
      <th>Weston County</th>
      <td>360.0</td>
      <td>3107.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7117.0</td>
      <td>0.527750</td>
      <td>1.4</td>
      <td>91.6</td>
      <td>0.5</td>
      <td>0.1</td>
      <td>4.3</td>
      <td>0.0</td>
      <td>0.774203</td>
      <td>30955.0</td>
      <td>14.4</td>
      <td>24.1</td>
      <td>32.0</td>
      <td>15.0</td>
      <td>15.8</td>
      <td>17.9</td>
      <td>19.3</td>
      <td>72.7</td>
      <td>6.7</td>
      <td>9.1</td>
      <td>3.0</td>
      <td>1.6</td>
      <td>6.9</td>
      <td>25.7</td>
      <td>0.478713</td>
      <td>68.2</td>
      <td>21.9</td>
      <td>8.8</td>
      <td>1.1</td>
      <td>2.2</td>
    </tr>
  </tbody>
</table>
<p>4809 rows × 34 columns</p>
</div>



- 카운티 지사를 선출하지 않는 county의 경우 NaN값으로 표시된다.

### 컬럼간 상관관계 살펴보기


```python
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.show()
```


![png](output_30_0.png)


- 전처리가 덜 되었음. 민주당을 뽑은 인원이 공화당에도 큰 영향을 미친다는 것은 인구자체에 영향을 받아 다중공선성이 일어나고 있다는 증거.
- 비율로 바꾸어 줄 필요가 있음.
- Asian 투표율이 높음.


```python
df_norm = df.copy()
df_norm['Pres_DEM'] /= df['Pres_DEM'] + df['Pres_REP']
df_norm['Pres_REP'] /= df['Pres_DEM'] + df['Pres_REP']
df_norm['Gov_DEM'] /= df['Gov_DEM'] + df['Gov_REP']
df_norm['Gov_REP'] /= df['Gov_DEM'] + df['Gov_REP']
```


```python
plt.figure(figsize=(5,10))
sns.heatmap(df_norm.corr()[['Pres_DEM', 'Pres_REP']], annot=True)
plt.show()
```


![png](output_33_0.png)


- 인구가 많은 county일수록 민주당 지지, 백인은 공화동 유색인종은민주당 
- 전문직,서비스직,사무직 민주당. 건설,생산,운송 공화당.


```python
sns.jointplot(data=df_norm, x='White', y='Pres_REP', kind='hex')
```




    <seaborn.axisgrid.JointGrid at 0x1bbd39a4f48>




![png](output_35_1.png)


- 단순히 백인비율이 많다고 공화당 지지가 높지는 않음. 아마 직업의 영향이 반영된듯 하다.


```python
sns.jointplot(data=df_norm, x='White', y='Pres_REP', hue='Professional')
```

    C:\Users\dissi\anaconda31\lib\site-packages\seaborn\distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)
    C:\Users\dissi\anaconda31\lib\site-packages\seaborn\distributions.py:306: UserWarning: Dataset has 0 variance; skipping density estimate.
      warnings.warn(msg, UserWarning)
    




    <seaborn.axisgrid.JointGrid at 0x1bbd3dd1f48>




![png](output_37_2.png)


- 아래로 내려갈수록(공화당 지지가 낮을수록) 전문직비율이 높음.


```python
sns.jointplot(data=df_norm, x='Black', y='Pres_DEM', alpha=0.2)
```




    <seaborn.axisgrid.JointGrid at 0x1bbd3934488>




![png](output_39_1.png)


- 상관성은 확실히 있으나 위의 plot과 마찬가지로 black 비율이 낮다고 민주당 지지도 낮지는 않음.
- 또한 흑인이 많은 county 자체도 많지는 않아 전체 데이터에 큰 영향을 주지는 않음. 

### Plotly로 시각화

#### 전처리


```python
import plotly.figure_factory as ff

df_sample = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/laucnty16.csv')
df_sample['State FIPS Code'] = df_sample['State FIPS Code'].apply(lambda x: str(x).zfill(2))
df_sample['County FIPS Code'] = df_sample['County FIPS Code'].apply(lambda x: str(x).zfill(3))
df_sample['FIPS'] = df_sample['State FIPS Code'] + df_sample['County FIPS Code']

colorscale = ["#f7fbff","#ebf3fb","#deebf7","#d2e3f3","#c6dbef","#b3d2e9","#9ecae1",
              "#85bcdb","#6baed6","#57a0ce","#4292c6","#3082be","#2171b5","#1361a9",
              "#08519c","#0b4083","#08306b"]
```


```python
df_sample
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
      <th>LAUS Code</th>
      <th>State FIPS Code</th>
      <th>County FIPS Code</th>
      <th>County Name/State Abbreviation</th>
      <th>Year</th>
      <th>Labor Force</th>
      <th>Employed</th>
      <th>Unemployed</th>
      <th>Unemployment Rate (%)</th>
      <th>FIPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CN0100100000000</td>
      <td>01</td>
      <td>001</td>
      <td>Autauga County, AL</td>
      <td>2016</td>
      <td>25,649</td>
      <td>24,297</td>
      <td>1,352</td>
      <td>5.3</td>
      <td>01001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CN0100300000000</td>
      <td>01</td>
      <td>003</td>
      <td>Baldwin County, AL</td>
      <td>2016</td>
      <td>89,931</td>
      <td>85,061</td>
      <td>4,870</td>
      <td>5.4</td>
      <td>01003</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CN0100500000000</td>
      <td>01</td>
      <td>005</td>
      <td>Barbour County, AL</td>
      <td>2016</td>
      <td>8,302</td>
      <td>7,584</td>
      <td>718</td>
      <td>8.6</td>
      <td>01005</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN0100700000000</td>
      <td>01</td>
      <td>007</td>
      <td>Bibb County, AL</td>
      <td>2016</td>
      <td>8,573</td>
      <td>8,004</td>
      <td>569</td>
      <td>6.6</td>
      <td>01007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CN0100900000000</td>
      <td>01</td>
      <td>009</td>
      <td>Blount County, AL</td>
      <td>2016</td>
      <td>24,525</td>
      <td>23,171</td>
      <td>1,354</td>
      <td>5.5</td>
      <td>01009</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3214</th>
      <td>CN7214500000000</td>
      <td>72</td>
      <td>145</td>
      <td>Vega Baja Municipio, PR</td>
      <td>2016</td>
      <td>13,812</td>
      <td>11,894</td>
      <td>1,918</td>
      <td>13.9</td>
      <td>72145</td>
    </tr>
    <tr>
      <th>3215</th>
      <td>CN7214700000000</td>
      <td>72</td>
      <td>147</td>
      <td>Vieques Municipio, PR</td>
      <td>2016</td>
      <td>3,287</td>
      <td>2,939</td>
      <td>348</td>
      <td>10.6</td>
      <td>72147</td>
    </tr>
    <tr>
      <th>3216</th>
      <td>CN7214900000000</td>
      <td>72</td>
      <td>149</td>
      <td>Villalba Municipio, PR</td>
      <td>2016</td>
      <td>7,860</td>
      <td>6,273</td>
      <td>1,587</td>
      <td>20.2</td>
      <td>72149</td>
    </tr>
    <tr>
      <th>3217</th>
      <td>CN7215100000000</td>
      <td>72</td>
      <td>151</td>
      <td>Yabucoa Municipio, PR</td>
      <td>2016</td>
      <td>9,137</td>
      <td>7,591</td>
      <td>1,546</td>
      <td>16.9</td>
      <td>72151</td>
    </tr>
    <tr>
      <th>3218</th>
      <td>CN7215300000000</td>
      <td>72</td>
      <td>153</td>
      <td>Yauco Municipio, PR</td>
      <td>2016</td>
      <td>10,815</td>
      <td>8,783</td>
      <td>2,032</td>
      <td>18.8</td>
      <td>72153</td>
    </tr>
  </tbody>
</table>
<p>3219 rows × 10 columns</p>
</div>




```python
state_code.head()
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
      <th>State/District</th>
      <th>Abbreviation</th>
      <th>Postal Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Alabama</td>
      <td>Ala.</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Alaska</td>
      <td>Alaska</td>
      <td>AK</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Arizona</td>
      <td>Ariz.</td>
      <td>AZ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Arkansas</td>
      <td>Ark.</td>
      <td>AR</td>
    </tr>
    <tr>
      <th>4</th>
      <td>California</td>
      <td>Calif.</td>
      <td>CA</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_map = state_code.set_index('State/District')['Postal Code']
```


```python
state_map
```




    State/District
    Alabama                 AL
    Alaska                  AK
    Arizona                 AZ
    Arkansas                AR
    California              CA
    Colorado                CO
    Connecticut             CT
    Delaware                DE
    District of Columbia    DC
    Florida                 FL
    Georgia                 GA
    Hawaii                  HI
    Idaho                   ID
    Illinois                IL
    Indiana                 IN
    Iowa                    IA
    Kansas                  KS
    Kentucky                KY
    Louisiana               LA
    Maine                   ME
    Maryland                MD
    Massachusetts           MA
    Michigan                MI
    Minnesota               MN
    Mississippi             MS
    Missouri                MO
    Montana                 MT
    Nebraska                NE
    Nevada                  NV
    New Hampshire           NH
    New Jersey              NJ
    New Mexico              NM
    New York                NY
    North Carolina          NC
    North Dakota            ND
    Ohio                    OH
    Oklahoma                OK
    Oregon                  OR
    Pennsylvania            PA
    Rhode Island            RI
    South Carolina          SC
    South Dakota            SD
    Tennessee               TN
    Texas                   TX
    Utah                    UT
    Vermont                 VT
    Virginia                VA
    Washington              WA
    West Virginia           WV
    Wisconsin               WI
    Wyoming                 WY
    Name: Postal Code, dtype: object




```python
counties = df_norm.reset_index()['county'] + ', ' + df_norm.reset_index()['state'].map(state_map)
```


```python
counties
```




    0          Autauga County, AL
    1          Baldwin County, AL
    2          Barbour County, AL
    3             Bibb County, AL
    4           Blount County, AL
                    ...          
    4804    Sweetwater County, WY
    4805         Teton County, WY
    4806         Uinta County, WY
    4807      Washakie County, WY
    4808        Weston County, WY
    Length: 4809, dtype: object




```python
counties_to_fips = df_sample.set_index('County Name/State Abbreviation')['FIPS']
counties_to_fips
```




    County Name/State Abbreviation
    Autauga County, AL         01001
    Baldwin County, AL         01003
    Barbour County, AL         01005
    Bibb County, AL            01007
    Blount County, AL          01009
                               ...  
    Vega Baja Municipio, PR    72145
    Vieques Municipio, PR      72147
    Villalba Municipio, PR     72149
    Yabucoa Municipio, PR      72151
    Yauco Municipio, PR        72153
    Name: FIPS, Length: 3219, dtype: object




```python
fips = counties.map(counties_to_fips)
fips

# df_norm 의 지리정보를 fips코드 하나로 바꿈.
```




    0       01001
    1       01003
    2       01005
    3       01007
    4       01009
            ...  
    4804    56037
    4805    56039
    4806    56041
    4807    56043
    4808    56045
    Length: 4809, dtype: object




```python
fips.isna().sum()
```




    1681




```python
data = df_norm.reset_index()['Pres_DEM'][fips.notna()]
fips = fips[fips.notna()]

# data는 민주당 지지율을 index로 새로 구성하되 fips가 null값이 아닌 번호를 골라서
# Fips코드는 choropleth를 사용하기 위해(가시화)
```

#### 시각화


```python
fig = ff.create_choropleth(
    fips=fips, values=data,
    show_state_data = False,
    colorscale=colorscale,
    binning_endpoints=list(np.linspace(0.0, 1.0, len(colorscale) - 2)),
    show_hover=True, centroid_marker={'opacity':0},
    asp = 2.9, title="USA by Voting for DEM President"
)

fig.layout.template = None
fig.show()
```


```python

```

## 전처리


```python
df_norm.columns
```




    Index(['Pres_DEM', 'Pres_REP', 'Gov_DEM', 'Gov_REP', 'TotalPop', 'Men',
           'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific',
           'VotingAgeCitizen', 'IncomePerCap', 'Poverty', 'ChildPoverty',
           'Professional', 'Service', 'Office', 'Construction', 'Production',
           'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome',
           'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 'SelfEmployed',
           'FamilyWork', 'Unemployment'],
          dtype='object')




```python
df_norm.dropna(inplace=True)
X = df_norm.drop(['Pres_DEM', 'Pres_REP', 'Gov_DEM', 'Gov_REP'], axis=1)
y = df_norm['Pres_DEM']
```


```python
# 수치형 데이터 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X= pd.DataFrame(data=X_scaled, index=X.index, columns=X.columns)
```


```python
X.head()
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
      <th></th>
      <th>TotalPop</th>
      <th>Men</th>
      <th>Hispanic</th>
      <th>White</th>
      <th>Black</th>
      <th>Native</th>
      <th>Asian</th>
      <th>Pacific</th>
      <th>VotingAgeCitizen</th>
      <th>IncomePerCap</th>
      <th>Poverty</th>
      <th>ChildPoverty</th>
      <th>Professional</th>
      <th>Service</th>
      <th>Office</th>
      <th>Construction</th>
      <th>Production</th>
      <th>Drive</th>
      <th>Carpool</th>
      <th>Transit</th>
      <th>Walk</th>
      <th>OtherTransp</th>
      <th>WorkAtHome</th>
      <th>MeanCommute</th>
      <th>Employed</th>
      <th>PrivateWork</th>
      <th>PublicWork</th>
      <th>SelfEmployed</th>
      <th>FamilyWork</th>
      <th>Unemployment</th>
    </tr>
    <tr>
      <th>state</th>
      <th>county</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">Delaware</th>
      <th>Kent County</th>
      <td>0.651462</td>
      <td>-0.946145</td>
      <td>0.346057</td>
      <td>-1.423723</td>
      <td>1.751813</td>
      <td>-0.191121</td>
      <td>0.783427</td>
      <td>-0.393531</td>
      <td>-0.388638</td>
      <td>0.313498</td>
      <td>-0.444312</td>
      <td>-0.300398</td>
      <td>0.327031</td>
      <td>0.447702</td>
      <td>0.504056</td>
      <td>-0.807226</td>
      <td>-0.379883</td>
      <td>0.447917</td>
      <td>-0.170797</td>
      <td>0.559713</td>
      <td>-0.455170</td>
      <td>-0.337372</td>
      <td>-0.325794</td>
      <td>0.608072</td>
      <td>0.209912</td>
      <td>-0.063847</td>
      <td>0.791294</td>
      <td>-0.836585</td>
      <td>-0.243581</td>
      <td>0.265911</td>
    </tr>
    <tr>
      <th>New Castle County</th>
      <td>3.043033</td>
      <td>-0.848175</td>
      <td>0.828313</td>
      <td>-1.695454</td>
      <td>1.751813</td>
      <td>-0.239442</td>
      <td>3.218819</td>
      <td>-0.393531</td>
      <td>-0.665857</td>
      <td>1.606755</td>
      <td>-0.638605</td>
      <td>-0.567309</td>
      <td>1.978537</td>
      <td>-0.315791</td>
      <td>0.658756</td>
      <td>-1.535077</td>
      <td>-1.212547</td>
      <td>0.221028</td>
      <td>-0.857051</td>
      <td>3.061686</td>
      <td>-0.301446</td>
      <td>-0.337372</td>
      <td>-0.325794</td>
      <td>0.492385</td>
      <td>0.801126</td>
      <td>1.168959</td>
      <td>-0.715528</td>
      <td>-0.999491</td>
      <td>-0.404976</td>
      <td>0.192397</td>
    </tr>
    <tr>
      <th>Sussex County</th>
      <td>0.917027</td>
      <td>-0.818852</td>
      <td>0.759420</td>
      <td>-0.634409</td>
      <td>0.622026</td>
      <td>-0.215282</td>
      <td>0.210394</td>
      <td>-0.393531</td>
      <td>0.227859</td>
      <td>1.122397</td>
      <td>-0.620942</td>
      <td>-0.044163</td>
      <td>0.295272</td>
      <td>0.147758</td>
      <td>0.844395</td>
      <td>-0.469295</td>
      <td>-0.543150</td>
      <td>0.632264</td>
      <td>-1.001526</td>
      <td>0.112932</td>
      <td>-0.506412</td>
      <td>-0.141175</td>
      <td>0.055883</td>
      <td>0.415259</td>
      <td>-0.026356</td>
      <td>0.503244</td>
      <td>-0.251891</td>
      <td>-0.470046</td>
      <td>-0.404976</td>
      <td>0.045370</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Indiana</th>
      <th>Adams County</th>
      <td>-0.213551</td>
      <td>-0.230718</td>
      <td>-0.067305</td>
      <td>0.581911</td>
      <td>-0.432442</td>
      <td>-0.263602</td>
      <td>-0.362639</td>
      <td>-0.393531</td>
      <td>-1.625095</td>
      <td>-0.755470</td>
      <td>0.597804</td>
      <td>0.916716</td>
      <td>-1.149795</td>
      <td>-0.615735</td>
      <td>-0.516958</td>
      <td>0.232561</td>
      <td>1.660960</td>
      <td>-0.658166</td>
      <td>2.393626</td>
      <td>-0.512562</td>
      <td>-0.250204</td>
      <td>0.447416</td>
      <td>-0.244006</td>
      <td>-0.066773</td>
      <td>-0.205611</td>
      <td>0.971710</td>
      <td>-1.344751</td>
      <td>-0.001691</td>
      <td>0.401999</td>
      <td>-0.285443</td>
    </tr>
    <tr>
      <th>Allen County</th>
      <td>1.870146</td>
      <td>-0.656809</td>
      <td>0.414951</td>
      <td>-0.653818</td>
      <td>0.546706</td>
      <td>-0.239442</td>
      <td>1.857865</td>
      <td>-0.393531</td>
      <td>-1.068441</td>
      <td>0.224871</td>
      <td>-0.144041</td>
      <td>0.083954</td>
      <td>0.358791</td>
      <td>-0.452129</td>
      <td>0.937214</td>
      <td>-1.405104</td>
      <td>0.289514</td>
      <td>0.759889</td>
      <td>-0.495865</td>
      <td>0.202288</td>
      <td>-0.608895</td>
      <td>-0.533569</td>
      <td>-0.380320</td>
      <td>-0.355993</td>
      <td>0.707837</td>
      <td>1.477161</td>
      <td>-1.278517</td>
      <td>-0.816222</td>
      <td>-0.404976</td>
      <td>0.118884</td>
    </tr>
  </tbody>
</table>
</div>



#### 학습, 테스트 데이터 분리


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

#### PCA 이용 (Columns 수가 많음.)


```python
from sklearn.decomposition import PCA
```


```python
pca = PCA()
pca.fit(X_train)
plt.plot(range(1, len(pca.explained_variance_) + 1), pca.explained_variance_)
plt.grid()

# 1부터 30개의 Columns. 10정도의 변수만 되어도 충분히 설명가능.
```


![png](output_66_0.png)



```python
pca = PCA(n_components=10)
pca.fit(X_train)
```




    PCA(copy=True, iterated_power='auto', n_components=10, random_state=None,
        svd_solver='auto', tol=0.0, whiten=False)



## 모델 적용

#### LightGBM 모델


```python
from lightgbm import LGBMRegressor
model_reg = LGBMRegressor()
model_reg.fit(pca.transform(X_train), y_train)
```




    LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                  importance_type='split', learning_rate=0.1, max_depth=-1,
                  min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
                  n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
                  random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
                  subsample=1.0, subsample_for_bin=200000, subsample_freq=0)




```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report
from math import sqrt
```


```python
pred = model_reg.predict(pca.transform(X_test))
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred)))
```

    0.06323869955871446
    0.08461257022993489
    


```python
print(classification_report(y_test > 0.5, pred > 0.5))

# y = Pres_DEM 0.5 이상일경우 True
```

                  precision    recall  f1-score   support
    
           False       0.95      0.96      0.96       146
            True       0.62      0.59      0.61        17
    
        accuracy                           0.92       163
       macro avg       0.79      0.77      0.78       163
    weighted avg       0.92      0.92      0.92       163
    
    

#### XGBoost 모델


```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train > 0.5)
```

    [23:07:34] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1,
                  objective='binary:logistic', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=1, subsample=1,
                  tree_method='exact', use_label_encoder=True,
                  validate_parameters=1, verbosity=None)




```python
plt.figure(figsize=(12,10))
plt.barh(x.columns, model_xgb.feature_importances_)
plt.show()

# 민주당 표에 영향을 미치는 것. 인구는 당연한 변수이므로 제외한다면 asian, transit, publicwork 눈여겨볼만하다.
```


![png](output_76_0.png)



```python
pred = model_xgb.predict(X_test)
print(classification_report(y_test>0.5 , pred))
```

                  precision    recall  f1-score   support
    
           False       0.96      0.97      0.97       146
            True       0.73      0.65      0.69        17
    
        accuracy                           0.94       163
       macro avg       0.85      0.81      0.83       163
    weighted avg       0.94      0.94      0.94       163
    
    


```python

```


```python

```


```python

```


```python

```
