---
layout: post
title:  "Kaggle - New York City Airbnb"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# New York Cty Airbnb Open Data

- 데이터 셋 :  https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

- 뉴욕시 에어비앤비에 전시된 여러 공간의 변수들을 통해 적당한 '이용료'를 파악-분류하고자 하는 데이터 셋이다.

## 라이브러리 설정 및 데이터 읽어들이기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('AB_NYC_2019.csv')

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
      <th>id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2539</td>
      <td>Clean &amp; quiet apt home by the park</td>
      <td>2787</td>
      <td>John</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>40.64749</td>
      <td>-73.97237</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2595</td>
      <td>Skylit Midtown Castle</td>
      <td>2845</td>
      <td>Jennifer</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>40.75362</td>
      <td>-73.98377</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3647</td>
      <td>THE VILLAGE OF HARLEM....NEW YORK !</td>
      <td>4632</td>
      <td>Elisabeth</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>40.80902</td>
      <td>-73.94190</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>40.68514</td>
      <td>-73.95976</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5022</td>
      <td>Entire Apt: Spacious Studio/Loft by central park</td>
      <td>7192</td>
      <td>Laura</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>40.79851</td>
      <td>-73.94399</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 48895 entries, 0 to 48894
    Data columns (total 16 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              48895 non-null  int64  
     1   name                            48879 non-null  object 
     2   host_id                         48895 non-null  int64  
     3   host_name                       48874 non-null  object 
     4   neighbourhood_group             48895 non-null  object 
     5   neighbourhood                   48895 non-null  object 
     6   latitude                        48895 non-null  float64
     7   longitude                       48895 non-null  float64
     8   room_type                       48895 non-null  object 
     9   price                           48895 non-null  int64  
     10  minimum_nights                  48895 non-null  int64  
     11  number_of_reviews               48895 non-null  int64  
     12  last_review                     38843 non-null  object 
     13  reviews_per_month               38843 non-null  float64
     14  calculated_host_listings_count  48895 non-null  int64  
     15  availability_365                48895 non-null  int64  
    dtypes: float64(3), int64(7), object(6)
    memory usage: 6.0+ MB



```python
df.isna().sum()
```




    id                                    0
    name                                 16
    host_id                               0
    host_name                            21
    neighbourhood_group                   0
    neighbourhood                         0
    latitude                              0
    longitude                             0
    room_type                             0
    price                                 0
    minimum_nights                        0
    number_of_reviews                     0
    last_review                       10052
    reviews_per_month                 10052
    calculated_host_listings_count        0
    availability_365                      0
    dtype: int64



- last_review 와 reviews_per_month 숫자가 같은것으로 보아 동일한 개수의 데이터일 듯 하다.


```python
(df['number_of_reviews'] == 0).sum()
```




    10052



- number of reviews가 0인 데이터가 10052인것으로 보아 해당 데이터가 last_review, reviews_per_month가 결측되어 있음을 알 수 있다.


```python
# 곱셉 기능을 통해 True False 유무 확인. & 기능을 통해 True False 확인.
(df['reviews_per_month'].isna() & df['last_review'].isna()).sum()
```




    10052



- 10052로 null 값 생성. 즉 두 변수가 가지고 있는 결측치의 인덱스가 동일

## EDA 및 기초통계 분석

### 불필요한 column 제거

- ID, host_name, latitude, longitude은 직관적으로 가격에 영향을 및지 않을 것 같음. 위도와 경도의 경우 다를 수 있겠으나 일단 제거.
- name 역시 자연어 처리를 통해 유의미한 결과값을 뽑아낼 수 있겠으나 일단 EDA - 데이터 분석을 위해 제거.
- 리뷰관련된 변수의 경우 리뷰의 유무라는 새로운 변수 생성가능. 이용 가능 일수 역시 0일(이용 일수 미입력)이라는 변수 새롭게 생성 가능


```python
df['room_type'].value_counts()
```




    Entire home/apt    25409
    Private room       22326
    Shared room         1160
    Name: room_type, dtype: int64




```python
df['availability_365'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x269799e7508>




![output_17_1](https://user-images.githubusercontent.com/77723966/112158851-e8abfd00-8c2b-11eb-9de1-1f05662e8684.png)



```python
# 이용 가능 일수가 0인 데이터. 

(df['availability_365'] == 0).sum()
```




    17533




```python
df.describe()
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
      <th>id</th>
      <th>host_id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.889500e+04</td>
      <td>4.889500e+04</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
      <td>38843.000000</td>
      <td>48895.000000</td>
      <td>48895.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.901714e+07</td>
      <td>6.762001e+07</td>
      <td>40.728949</td>
      <td>-73.952170</td>
      <td>152.720687</td>
      <td>7.029962</td>
      <td>23.274466</td>
      <td>1.373221</td>
      <td>7.143982</td>
      <td>112.781327</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.098311e+07</td>
      <td>7.861097e+07</td>
      <td>0.054530</td>
      <td>0.046157</td>
      <td>240.154170</td>
      <td>20.510550</td>
      <td>44.550582</td>
      <td>1.680442</td>
      <td>32.952519</td>
      <td>131.622289</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.539000e+03</td>
      <td>2.438000e+03</td>
      <td>40.499790</td>
      <td>-74.244420</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.471945e+06</td>
      <td>7.822033e+06</td>
      <td>40.690100</td>
      <td>-73.983070</td>
      <td>69.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.190000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.967728e+07</td>
      <td>3.079382e+07</td>
      <td>40.723070</td>
      <td>-73.955680</td>
      <td>106.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>0.720000</td>
      <td>1.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.915218e+07</td>
      <td>1.074344e+08</td>
      <td>40.763115</td>
      <td>-73.936275</td>
      <td>175.000000</td>
      <td>5.000000</td>
      <td>24.000000</td>
      <td>2.020000</td>
      <td>2.000000</td>
      <td>227.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.648724e+07</td>
      <td>2.743213e+08</td>
      <td>40.913060</td>
      <td>-73.712990</td>
      <td>10000.000000</td>
      <td>1250.000000</td>
      <td>629.000000</td>
      <td>58.500000</td>
      <td>327.000000</td>
      <td>365.000000</td>
    </tr>
  </tbody>
</table>

</div>



- 가격의 경우 최소가격과 최대가격의 설정이 잘못 되었다. 최소 숙박일수 역시 최대값이 잘못 설정되었다.


```python
df.columns
```




    Index(['id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
           'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
           'minimum_nights', 'number_of_reviews', 'last_review',
           'reviews_per_month', 'calculated_host_listings_count',
           'availability_365'],
          dtype='object')




```python
df.drop(['id', 'name', 'host_name', 'latitude', 'longitude'], axis=1, inplace=True)
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
      <th>host_id</th>
      <th>neighbourhood_group</th>
      <th>neighbourhood</th>
      <th>room_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2787</td>
      <td>Brooklyn</td>
      <td>Kensington</td>
      <td>Private room</td>
      <td>149</td>
      <td>1</td>
      <td>9</td>
      <td>2018-10-19</td>
      <td>0.21</td>
      <td>6</td>
      <td>365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2845</td>
      <td>Manhattan</td>
      <td>Midtown</td>
      <td>Entire home/apt</td>
      <td>225</td>
      <td>1</td>
      <td>45</td>
      <td>2019-05-21</td>
      <td>0.38</td>
      <td>2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4632</td>
      <td>Manhattan</td>
      <td>Harlem</td>
      <td>Private room</td>
      <td>150</td>
      <td>3</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1</td>
      <td>365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4869</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>Entire home/apt</td>
      <td>89</td>
      <td>1</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>1</td>
      <td>194</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7192</td>
      <td>Manhattan</td>
      <td>East Harlem</td>
      <td>Entire home/apt</td>
      <td>80</td>
      <td>10</td>
      <td>9</td>
      <td>2018-11-19</td>
      <td>0.10</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>



### 수치형 데이터

- 수치형 데이터를 통해서만 price를 예측할 수 있을까?


```python
sns.jointplot(data=df, x='reviews_per_month', y='price')
```




    <seaborn.axisgrid.JointGrid at 0x2697ab33148>




![output_26_1](https://user-images.githubusercontent.com/77723966/112158889-f2356500-8c2b-11eb-8d46-e370453ed2a1.png)




```python
sns.heatmap(df.corr(), annot=True, cmap='YlOrRd')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2697b48fec8>




![output_27_1](https://user-images.githubusercontent.com/77723966/112158905-f5c8ec00-8c2b-11eb-9278-f972a5e8d64a.png)



- 전처리가 되어 있지 않아서 상관성을 파악하기 어렵다.
- 의외로 host_id가 상관성이 있어보이는데 사용일수와 0.2, 한달동안 리뷰는 0.3 으로 확인할 수 있다.
- 새롭게 운영하는 사람일수록 큰 숫자의 id를 가지고 있고 새 시설인 만큼 리뷰도 많이 받고 오래 이용할 수 있기 때문인지 확인이 더 필요하다.
- host_id에 있어서 누적리뷰와 한달동안 받는 리뷰가 다른 상관성을 띈다.

### 범주형 데이터


```python
sns.boxplot(data=df, x='neighbourhood_group', y='price')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2697e22af08>



![output_30_1](https://user-images.githubusercontent.com/77723966/112158925-f9f50980-8c2b-11eb-9729-b105a78056d1.png)



- 최대값이외에도 평균 기준값을 훨씬 초과하는 아웃라이어 price 값들이 있어서 확인하기 어렵다. 전처리 필요

## 데이터 전처리

### 범주형 데이터 전처리


```python
df['neighbourhood_group'].value_counts()
```




    Manhattan        21661
    Brooklyn         20104
    Queens            5666
    Bronx             1091
    Staten Island      373
    Name: neighbourhood_group, dtype: int64




```python
df['neighbourhood'].value_counts()
```




    Williamsburg          3920
    Bedford-Stuyvesant    3714
    Harlem                2658
    Bushwick              2465
    Upper West Side       1971
                          ... 
    Woodrow                  1
    Fort Wadsworth           1
    New Dorp                 1
    Richmondtown             1
    Rossville                1
    Name: neighbourhood, Length: 221, dtype: int64




```python
plt.plot(range(len(df['neighbourhood'].value_counts())), df['neighbourhood'].value_counts())
```




    [<matplotlib.lines.Line2D at 0x2697e315588>]




![output_36_1](https://user-images.githubusercontent.com/77723966/112158936-fe212700-8c2b-11eb-9749-d8771a864ae1.png)



- 소수의 값 제거를 위해 50번째 이전 value 값만 잔존시킴.


```python
ne = df['neighbourhood'].value_counts()[50:]
```


```python
df['neighbourhood'] = df['neighbourhood'].apply(lambda s : s if str(s) not in ne[50:] else 'others')
```

### 수치형 데이터 전처리


```python
sns.rugplot(data=df, x='price', height=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2697e36a348>




![output_41_1](https://user-images.githubusercontent.com/77723966/112158961-01b4ae00-8c2c-11eb-9519-45262c3c110f.png)




```python
print(df['price'].quantile(0.99))
print(df['price'].quantile(0.005))
```

    799.0
    26.0


- price값 상위 1% 가 799달러 즉, 1000달러 이후의 데이터는 아웃라이어 값으로 판단할 수 있다.
- price값 하위 아웃라이어 값 제거를 위해 0.5% 제거, 상위는 5% 제거


```python
sns.rugplot(data=df, x='minimum_nights', height=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2697e4aaf88>



![output_44_1](https://user-images.githubusercontent.com/77723966/112158979-05483500-8c2c-11eb-90c2-9f0436d8e0c4.png)



```python
print(df['minimum_nights'].quantile(0.98))
print(df['minimum_nights'].quantile(0.005))
```

    30.0
    1.0


- minimum_nights 경우 최소값은 자를 필요가 없고 상위 2%정도로 자르면 될 듯 하다.


```python
sns.rugplot(data=df, x='availability_365', height=1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x269028d0e88>




![output_47_1](https://user-images.githubusercontent.com/77723966/112158989-08432580-8c2c-11eb-8b92-83a1dddd0099.png)



```python
df['availability_365'].quantile(0.3)
```




    0.0



- 30% 까지 0이라는 것은 미입력된 데이터가 상당하다는 뜻으로 차라리 이용일수가'0'인 새로운 범주형변수를 생성하는것이 효율적으로 보인다.


```python
p1= df['price'].quantile(0.95)
p2= df['price'].quantile(0.005)

df = df[(p1 > df['price']) & (df['price'] > p2)]
```


```python
df['price'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x269042e5fc8>




![output_51_1](https://user-images.githubusercontent.com/77723966/112159006-0bd6ac80-8c2c-11eb-8365-b1b580c5327c.png)



```python
df['minimum_nights'].quantile(0.98)
```




    30.0



- price에서 아웃라이어 제거해도 상위 2%값 변하지 않음. 그대로 진행해도 좋다.


```python
m1 = df['minimum_nights'].quantile(0.98)
df = df[df['minimum_nights'] < m1 ]
```


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_num = df.drop(['neighbourhood_group', 'neighbourhood', 'room_type', 'avail_zero', 'review_exists','last_review','price'], axis=1)

scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

# last_review 경우 수치형이긴 하지만 날짜형태
```

### 범주형 데이터 전처리


```python
# availability_365 중 새로운 범주형 변수 생성

df['avail_zero'] = df['availability_365'].apply(lambda x : 'Zero' if x==0 else 'NonZero')
```


```python
# review 중 Null 값 채우기.

df['review_exists'] = df['reviews_per_month'].isna().apply(lambda x: 'No' if x is True else 'Yes')

df.fillna(0, inplace=True) # 현재 미기입은 리뷰만 있음.
```


```python
df.columns
```




    Index(['host_id', 'neighbourhood_group', 'neighbourhood', 'room_type', 'price',
           'minimum_nights', 'number_of_reviews', 'last_review',
           'reviews_per_month', 'calculated_host_listings_count',
           'availability_365', 'avail_zero', 'review_exists'],
          dtype='object')




```python
X_cat = df[['neighbourhood_group', 'neighbourhood', 'room_type', 'avail_zero', 'review_exists']]
X_cat = pd.get_dummies(X_cat)

# 선형회귀일 경우 get dummies 실행시 drop_first 그라디언트 부스트 사용시 X.
```


```python
X = pd.concat([X_scaled, X_cat], axis=1)
y = df['price']
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
      <th>host_id</th>
      <th>minimum_nights</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
      <th>calculated_host_listings_count</th>
      <th>availability_365</th>
      <th>neighbourhood_group_Bronx</th>
      <th>neighbourhood_group_Brooklyn</th>
      <th>neighbourhood_group_Manhattan</th>
      <th>neighbourhood_group_Queens</th>
      <th>neighbourhood_group_Staten Island</th>
      <th>neighbourhood_Arverne</th>
      <th>neighbourhood_Astoria</th>
      <th>neighbourhood_Battery Park City</th>
      <th>neighbourhood_Bay Ridge</th>
      <th>neighbourhood_Bedford-Stuyvesant</th>
      <th>neighbourhood_Bensonhurst</th>
      <th>neighbourhood_Boerum Hill</th>
      <th>neighbourhood_Borough Park</th>
      <th>neighbourhood_Briarwood</th>
      <th>neighbourhood_Brighton Beach</th>
      <th>neighbourhood_Brooklyn Heights</th>
      <th>neighbourhood_Brownsville</th>
      <th>neighbourhood_Bushwick</th>
      <th>neighbourhood_Canarsie</th>
      <th>neighbourhood_Carroll Gardens</th>
      <th>neighbourhood_Chelsea</th>
      <th>neighbourhood_Chinatown</th>
      <th>neighbourhood_Civic Center</th>
      <th>neighbourhood_Clinton Hill</th>
      <th>neighbourhood_Cobble Hill</th>
      <th>neighbourhood_Concourse</th>
      <th>neighbourhood_Corona</th>
      <th>neighbourhood_Crown Heights</th>
      <th>neighbourhood_Cypress Hills</th>
      <th>neighbourhood_Ditmars Steinway</th>
      <th>neighbourhood_Downtown Brooklyn</th>
      <th>neighbourhood_East Elmhurst</th>
      <th>neighbourhood_East Flatbush</th>
      <th>neighbourhood_East Harlem</th>
      <th>neighbourhood_East New York</th>
      <th>neighbourhood_East Village</th>
      <th>neighbourhood_Elmhurst</th>
      <th>neighbourhood_Financial District</th>
      <th>neighbourhood_Flatbush</th>
      <th>neighbourhood_Flatiron District</th>
      <th>neighbourhood_Flatlands</th>
      <th>neighbourhood_Flushing</th>
      <th>neighbourhood_Fordham</th>
      <th>neighbourhood_Forest Hills</th>
      <th>neighbourhood_Fort Greene</th>
      <th>neighbourhood_Fort Hamilton</th>
      <th>neighbourhood_Glendale</th>
      <th>neighbourhood_Gowanus</th>
      <th>neighbourhood_Gramercy</th>
      <th>neighbourhood_Gravesend</th>
      <th>neighbourhood_Greenpoint</th>
      <th>neighbourhood_Greenwich Village</th>
      <th>neighbourhood_Harlem</th>
      <th>neighbourhood_Hell's Kitchen</th>
      <th>neighbourhood_Inwood</th>
      <th>neighbourhood_Jackson Heights</th>
      <th>neighbourhood_Jamaica</th>
      <th>neighbourhood_Kensington</th>
      <th>neighbourhood_Kingsbridge</th>
      <th>neighbourhood_Kips Bay</th>
      <th>neighbourhood_Little Italy</th>
      <th>neighbourhood_Long Island City</th>
      <th>neighbourhood_Longwood</th>
      <th>neighbourhood_Lower East Side</th>
      <th>neighbourhood_Maspeth</th>
      <th>neighbourhood_Midtown</th>
      <th>neighbourhood_Midwood</th>
      <th>neighbourhood_Morningside Heights</th>
      <th>neighbourhood_Mott Haven</th>
      <th>neighbourhood_Murray Hill</th>
      <th>neighbourhood_NoHo</th>
      <th>neighbourhood_Nolita</th>
      <th>neighbourhood_Ozone Park</th>
      <th>neighbourhood_Park Slope</th>
      <th>neighbourhood_Port Morris</th>
      <th>neighbourhood_Prospect Heights</th>
      <th>neighbourhood_Prospect-Lefferts Gardens</th>
      <th>neighbourhood_Queens Village</th>
      <th>neighbourhood_Red Hook</th>
      <th>neighbourhood_Rego Park</th>
      <th>neighbourhood_Richmond Hill</th>
      <th>neighbourhood_Ridgewood</th>
      <th>neighbourhood_Rockaway Beach</th>
      <th>neighbourhood_Roosevelt Island</th>
      <th>neighbourhood_Rosedale</th>
      <th>neighbourhood_Sheepshead Bay</th>
      <th>neighbourhood_SoHo</th>
      <th>neighbourhood_South Slope</th>
      <th>neighbourhood_Springfield Gardens</th>
      <th>neighbourhood_St. Albans</th>
      <th>neighbourhood_St. George</th>
      <th>neighbourhood_Sunnyside</th>
      <th>neighbourhood_Sunset Park</th>
      <th>neighbourhood_Theater District</th>
      <th>neighbourhood_Tribeca</th>
      <th>neighbourhood_Two Bridges</th>
      <th>neighbourhood_Upper East Side</th>
      <th>neighbourhood_Upper West Side</th>
      <th>neighbourhood_Wakefield</th>
      <th>neighbourhood_Washington Heights</th>
      <th>neighbourhood_West Village</th>
      <th>neighbourhood_Williamsburg</th>
      <th>neighbourhood_Windsor Terrace</th>
      <th>neighbourhood_Woodhaven</th>
      <th>neighbourhood_Woodside</th>
      <th>neighbourhood_others</th>
      <th>room_type_Entire home/apt</th>
      <th>room_type_Private room</th>
      <th>room_type_Shared room</th>
      <th>avail_zero_NonZero</th>
      <th>avail_zero_Zero</th>
      <th>review_exists_No</th>
      <th>review_exists_Yes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.848227</td>
      <td>-0.588004</td>
      <td>-0.352358</td>
      <td>-0.588299</td>
      <td>0.066069</td>
      <td>2.132586</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.848227</td>
      <td>-0.588004</td>
      <td>0.419649</td>
      <td>-0.485721</td>
      <td>-0.084134</td>
      <td>2.052616</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.848204</td>
      <td>-0.119342</td>
      <td>-0.545360</td>
      <td>-0.715013</td>
      <td>-0.121684</td>
      <td>2.132586</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.848200</td>
      <td>-0.588004</td>
      <td>5.244692</td>
      <td>2.084766</td>
      <td>-0.121684</td>
      <td>0.765095</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.848170</td>
      <td>1.520973</td>
      <td>-0.352358</td>
      <td>-0.654673</td>
      <td>-0.121684</td>
      <td>-0.786327</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



### 학습 및 테스트 데이터 분리


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 분류하기 -  XGBoost Regression 모델 적용


```python
from xgboost import XGBRegressor

model_reg =XGBRegressor()
model_reg.fit(X_train, y_train)
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



### 모델 평가


```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

pred = model_reg.predict(X_test)
print(mean_absolute_error(y_test, pred))
print(sqrt(mean_squared_error(y_test, pred)))
```

    34.158105898456235
    48.339277624918665


- 낮은 값인지 높은값인지 직관적으로 쉬이 알기 어렵다.


```python
plt.scatter(x=y_test, y=pred, alpha=0.1)
plt.plot([0,350], [0,350], 'r-')
```




    [<matplotlib.lines.Line2D at 0x26907658148>]




![output_70_1](https://user-images.githubusercontent.com/77723966/112159053-15601480-8c2c-11eb-82fb-a05281f289f3.png)


- price 값의 경우 단위별로 나누어져 있는 경우가 많아 구간별로 나누어진다.
- 값이 낮을때는 overestimate 높을 경우는 underestimate 하는 현상이 일어남.


```python
err = (pred - y_test) / y_test
sns.histplot(err)
plt.grid()
```

![output_72_0](https://user-images.githubusercontent.com/77723966/112159069-185b0500-8c2c-11eb-9a9b-0ce744f97a5e.png)

