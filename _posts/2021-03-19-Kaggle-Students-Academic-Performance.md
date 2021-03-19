---
layout: post
title:  "Kaggle - Student's Academic Performance"
description: "캐글 데이터 분석"
author: SeungRok OH
categories: [Kaggle]
---

# Studnet's Academic Performance

- 데이터 셋 : https://www.kaggle.com/aljarah/xAPI-Edu-Data

- 학생들의 인적사항과 평가데이터를 통해 학생들의 성적을 예측하고자 하는 데이터셋이다.

## 라이브러리 설정 및 데이터 불러오기


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/dissi/Kaggle Practice/xAPI-Edu-Data.csv')
```


```python
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
      <th>gender</th>
      <th>NationalITy</th>
      <th>PlaceofBirth</th>
      <th>StageID</th>
      <th>GradeID</th>
      <th>SectionID</th>
      <th>Topic</th>
      <th>Semester</th>
      <th>Relation</th>
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>ParentAnsweringSurvey</th>
      <th>ParentschoolSatisfaction</th>
      <th>StudentAbsenceDays</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>453</th>
      <td>F</td>
      <td>Jordan</td>
      <td>Jordan</td>
      <td>MiddleSchool</td>
      <td>G-08</td>
      <td>A</td>
      <td>Geology</td>
      <td>S</td>
      <td>Father</td>
      <td>29</td>
      <td>78</td>
      <td>40</td>
      <td>12</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Above-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>412</th>
      <td>M</td>
      <td>Palestine</td>
      <td>Jordan</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>B</td>
      <td>Biology</td>
      <td>F</td>
      <td>Father</td>
      <td>78</td>
      <td>80</td>
      <td>66</td>
      <td>51</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>339</th>
      <td>F</td>
      <td>Palestine</td>
      <td>Jordan</td>
      <td>lowerlevel</td>
      <td>G-02</td>
      <td>B</td>
      <td>French</td>
      <td>S</td>
      <td>Father</td>
      <td>79</td>
      <td>89</td>
      <td>11</td>
      <td>14</td>
      <td>No</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>55</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>A</td>
      <td>Math</td>
      <td>F</td>
      <td>Father</td>
      <td>16</td>
      <td>14</td>
      <td>6</td>
      <td>20</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>471</th>
      <td>M</td>
      <td>Palestine</td>
      <td>Jordan</td>
      <td>MiddleSchool</td>
      <td>G-08</td>
      <td>A</td>
      <td>History</td>
      <td>S</td>
      <td>Father</td>
      <td>78</td>
      <td>82</td>
      <td>78</td>
      <td>53</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>140</th>
      <td>M</td>
      <td>Tunis</td>
      <td>Tunis</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>A</td>
      <td>Quran</td>
      <td>F</td>
      <td>Father</td>
      <td>10</td>
      <td>60</td>
      <td>5</td>
      <td>20</td>
      <td>Yes</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>327</th>
      <td>M</td>
      <td>Jordan</td>
      <td>Jordan</td>
      <td>lowerlevel</td>
      <td>G-02</td>
      <td>A</td>
      <td>French</td>
      <td>S</td>
      <td>Father</td>
      <td>30</td>
      <td>10</td>
      <td>20</td>
      <td>5</td>
      <td>No</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>228</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>HighSchool</td>
      <td>G-11</td>
      <td>B</td>
      <td>Math</td>
      <td>S</td>
      <td>Mum</td>
      <td>73</td>
      <td>84</td>
      <td>77</td>
      <td>81</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Above-7</td>
      <td>H</td>
    </tr>
    <tr>
      <th>418</th>
      <td>M</td>
      <td>Palestine</td>
      <td>Jordan</td>
      <td>MiddleSchool</td>
      <td>G-07</td>
      <td>B</td>
      <td>Biology</td>
      <td>F</td>
      <td>Father</td>
      <td>88</td>
      <td>90</td>
      <td>76</td>
      <td>81</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>H</td>
    </tr>
    <tr>
      <th>378</th>
      <td>M</td>
      <td>Jordan</td>
      <td>Jordan</td>
      <td>lowerlevel</td>
      <td>G-02</td>
      <td>B</td>
      <td>Arabic</td>
      <td>F</td>
      <td>Father</td>
      <td>10</td>
      <td>30</td>
      <td>50</td>
      <td>91</td>
      <td>Yes</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 480 entries, 0 to 479
    Data columns (total 17 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   gender                    480 non-null    object
     1   NationalITy               480 non-null    object
     2   PlaceofBirth              480 non-null    object
     3   StageID                   480 non-null    object
     4   GradeID                   480 non-null    object
     5   SectionID                 480 non-null    object
     6   Topic                     480 non-null    object
     7   Semester                  480 non-null    object
     8   Relation                  480 non-null    object
     9   raisedhands               480 non-null    int64 
     10  VisITedResources          480 non-null    int64 
     11  AnnouncementsView         480 non-null    int64 
     12  Discussion                480 non-null    int64 
     13  ParentAnsweringSurvey     480 non-null    object
     14  ParentschoolSatisfaction  480 non-null    object
     15  StudentAbsenceDays        480 non-null    object
     16  Class                     480 non-null    object
    dtypes: int64(4), object(13)
    memory usage: 63.9+ KB
    

- gender : 학생 성별 
- NationalITy : 학생 국적
- PlaceofBirth : 학생이 태어난 국가
- StageID : 학생이 다니는 학교 (초, 중, 고)
- GradeID : 학생이 속한 성적 등급
- SectionID : 학생이 속한 반 이름
- Topic : 수강한 과목
- Semester : 수강한 학기 (1학기/2학기)
- Relatioin : 주 보호자와 학생의 관계
- raisedhands : 학생이 수업 중 손을 든 횟수
- VisITedReseources : 학생이 과목 교과과정을 확인한 횟수
- Discussion : 학생이 토론 그룹에 참여한 횟수
- AnnouncementsView : 학생이 공지를 확인한 횟수
- ParentAnsweringSurvey : 부모가 학교 설문에 참여했는지 여부
- ParentschoolSatisfaction : 부모가 학교에 만족했는지 여부
- StudentAbsenceDays : 학생 결석 횟수 (7회 이상/ 미만)
- Class : 학생 성적 등급 (L 낮음, M 보통, H 높음)



```python
df['NationalITy'].value_counts()
```




    KW             179
    Jordan         172
    Palestine       28
    Iraq            22
    lebanon         17
    Tunis           12
    SaudiArabia     11
    Egypt            9
    Syria            7
    Lybia            6
    USA              6
    Iran             6
    Morocco          4
    venzuela         1
    Name: NationalITy, dtype: int64




```python
df['PlaceofBirth'].unique()
```




    array(['KuwaIT', 'lebanon', 'Egypt', 'SaudiArabia', 'USA', 'Jordan',
           'venzuela', 'Iran', 'Tunis', 'Morocco', 'Syria', 'Iraq',
           'Palestine', 'Lybia'], dtype=object)



## EDA 및 기초 통계 분석

### 수치형 데이터

- raisedhands, visitedresources, announcementsview, discussion


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
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>46.775000</td>
      <td>54.797917</td>
      <td>37.918750</td>
      <td>43.283333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.779223</td>
      <td>33.080007</td>
      <td>26.611244</td>
      <td>27.637735</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.750000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>65.000000</td>
      <td>33.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.000000</td>
      <td>84.000000</td>
      <td>58.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>99.000000</td>
      <td>98.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>
</div>



- 요약을 통해 보았을때 전체적으로 균형이 맞는 데이터라 볼 수 있다.


```python
sns.histplot(data=df, x='raisedhands',hue='Class', hue_order=['L', 'M', 'H'], kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661d411708>




![png](output_15_1.png)


- 손을 드는 횟수는 적거나 많거나 양쪽으로 나뉘어 나타나는데 Class 구분을 잘 반영하고 있다. 다만 수업시간에 손을 적게들어도, 많이들어도 중위권에 들어간 학생이 있는것으로 보아 완벽히 구분해내지는 못함을 보여준다.


```python
sns.histplot(data=df, x='VisITedResources',hue='Class', hue_order=['L', 'M', 'H'], kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661dc92108>




![png](output_17_1.png)


- raisedhands와 비슷한 양상으로 역시 Class 구분을 잘 반영하고 있다. 


```python
sns.histplot(data=df, x='AnnouncementsView',hue='Class', hue_order=['L', 'M', 'H'], kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661dd81948>




![png](output_19_1.png)


- 성적이 낮은 학생들의 구분은 쉽지만 중위권과 상위권 학생들의 구분이 모호함. 


```python
sns.histplot(data=df, x='Discussion',hue='Class', hue_order=['L', 'M', 'H'], kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661de54548>




![png](output_21_1.png)


- 다른 지표 대비 특별한 경향성이 보이지 않음.(하위권 학생들도 참여율이 나름 있고, 상위권 학생들도 두 양상으로 나타난다.)


```python
# raisedhands와 visitedresources의 경우 잘 나누어주고 있기 때문에 jointplot으로 함께 확인해본다.
sns.jointplot(data=df, x='VisITedResources', y='raisedhands', hue='Class', hue_order=['L','M',"H"])
```




    <seaborn.axisgrid.JointGrid at 0x1661df00488>




![png](output_23_1.png)


- 중위권과 상위권 구분은 여전히 어렵지만 하위권과 중위권은 jointplot, 2차원으로 확인시 더 분류할 수 있다..


```python
sns.pairplot(df, hue='Class', hue_order=['L','M','H'])
```




    <seaborn.axisgrid.PairGrid at 0x1661e05be08>




![png](output_25_1.png)


### 범주형 데이터


```python
sns.countplot(data=df, x='Class', order=['L', 'M', 'H']) 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661e960dc8>




![png](output_27_1.png)



```python
sns.countplot(data=df, x='gender', hue='Class', hue_order=['L', 'M', 'H'])

# 남녀 카테고리에 따른 성적 비교
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661fbad688>




![png](output_28_1.png)



```python
sns.countplot(data=df, x='NationalITy', hue='Class', hue_order=['L', 'M', 'H'])
plt.xticks(rotation=90)
plt.show()

# 국적에 따른 성적 비교
```


![png](output_29_0.png)



```python
sns.countplot(data=df, x='ParentAnsweringSurvey', hue='Class', hue_order=['L', 'M', 'H'])

# 부모 응답에 따른 성적 비교 
# 학교만족도의 경우 성적과 연관 가능성이 높으므로 빼는게 좋다고 판단 됨.
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1661eac8788>




![png](output_30_1.png)



```python
sns.countplot(data=df, x='Topic', hue='Class', hue_order=['L', 'M', 'H'])
plt.xticks(rotation=90)
plt.show()

# 과목에 따른 성적 비교 // 어떤 과목이 어려운지 
```


![png](output_31_0.png)


### 범주형 대상 Class을 수치로 바꾸어 표현
- 비율 파악을 위해, Low를 -1로, Middle을 0으로, High를 1로


```python
df['Class_value'] = df['Class'].map(dict(L=-1, M=0, H=1))
```


```python
gb_gender = df.groupby('gender').mean()['Class_value']
gb_gender
```




    gender
    F    0.291429
    M   -0.118033
    Name: Class_value, dtype: float64




```python
plt.bar(gb_gender.index, gb_gender)
```




    <BarContainer object of 2 artists>




![png](output_35_1.png)



```python
gb_Topic = df.groupby('Topic').mean()['Class_value'].sort_values()
plt.barh(gb_Topic.index, gb_Topic)
```




    <BarContainer object of 12 artists>




![png](output_36_1.png)


## 데이터 전처리

### 범주형 데이터를 one-hot vector로 변환


```python
# 컴퓨터는 0,1 밖에 인식할 수 없기 때문.

df.columns
```




    Index(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays', 'Class', 'Class_value'],
          dtype='object')




```python
# 다중공선성을 줄이기위해 drop_first를 True로
# drop을 써주지 않으면 해당칼럼을 수치형으로 인식하여 그대로 가져오게 됨. 빼고 싶으면 미표기가 아닌 drop를 써야 함.

X = pd.get_dummies(df.drop(['ParentschoolSatisfaction', 'Class', 'Class_value'], axis=1),
                   columns=['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
                            'SectionID', 'Topic', 'Semester', 'Relation','ParentAnsweringSurvey',
                            'StudentAbsenceDays'], drop_first=True)

y = df['Class']
```


```python
X.tail()
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
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>gender_M</th>
      <th>NationalITy_Iran</th>
      <th>NationalITy_Iraq</th>
      <th>NationalITy_Jordan</th>
      <th>NationalITy_KW</th>
      <th>NationalITy_Lybia</th>
      <th>...</th>
      <th>Topic_History</th>
      <th>Topic_IT</th>
      <th>Topic_Math</th>
      <th>Topic_Quran</th>
      <th>Topic_Science</th>
      <th>Topic_Spanish</th>
      <th>Semester_S</th>
      <th>Relation_Mum</th>
      <th>ParentAnsweringSurvey_Yes</th>
      <th>StudentAbsenceDays_Under-7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>475</th>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>476</th>
      <td>50</td>
      <td>77</td>
      <td>14</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>477</th>
      <td>55</td>
      <td>74</td>
      <td>25</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>478</th>
      <td>30</td>
      <td>17</td>
      <td>14</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>479</th>
      <td>35</td>
      <td>14</td>
      <td>23</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>



### 학습데이터 테스트 데이터 분리


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 모델 학습 및 평가

### Logistic Regression 모델 학습


```python
from sklearn.linear_model import LogisticRegression

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
```

    C:\Users\dissi\anaconda31\lib\site-packages\sklearn\linear_model\_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)
    




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=1000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# 평가

from sklearn.metrics import classification_report
pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               H       0.77      0.67      0.72        55
               L       0.78      0.76      0.77        33
               M       0.59      0.68      0.63        56
    
        accuracy                           0.69       144
       macro avg       0.72      0.70      0.71       144
    weighted avg       0.70      0.69      0.70       144
    
    

### XGBoost 모델 학습


```python
from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)
```

    C:\Users\dissi\anaconda31\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [21:42:35] WARNING: C:/Users/Administrator/workspace/xgboost-win64_release_1.3.0/src/learner.cc:1061: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
    




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=100, n_jobs=4, num_parallel_tree=1,
                  objective='multi:softprob', random_state=0, reg_alpha=0,
                  reg_lambda=1, scale_pos_weight=None, subsample=1,
                  tree_method='exact', use_label_encoder=True,
                  validate_parameters=1, verbosity=None)




```python
# 평가

pred = model_xgb.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               H       0.79      0.69      0.74        55
               L       0.85      0.85      0.85        33
               M       0.65      0.73      0.69        56
    
        accuracy                           0.74       144
       macro avg       0.76      0.76      0.76       144
    weighted avg       0.75      0.74      0.74       144
    
    

## 결과 분석


```python
model_lr.classes_
```




    array(['H', 'L', 'M'], dtype=object)




```python
# 회귀분석 결과

plt.figure(figsize=(15, 10))
plt.bar(X.columns, model_lr.coef_[0, :]) # H Class에 관여하는 요소들의 영향 정도
plt.xticks(rotation=90)
plt.show()
```


![png](output_53_0.png)


- 결석일수 7일 미만, 부모응답이 높고, 보호자가 어머니이며, 사우디아라비아 국적,출생이며 수학을 선택하면 
  성적이 높게 나온다.


```python
# 회귀분석 결과(성적을 낮게 하는 요소)
plt.figure(figsize=(15, 10))
plt.bar(X.columns, model_lr.coef_[1, :]) # L Class
plt.xticks(rotation=90)
plt.show()
```


![png](output_55_0.png)



```python
# xgboost분석 결과

plt.figure(figsize=(15, 10))
plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()
```


![png](output_56_0.png)


- 결석일수 7일 미만, 보호자가 어머니이며, 손을 많이 들고 공지를 많이 확인 할 경우 성적이 높음.


```python

```


```python

```
