# Data annotation
## Members

김한성|염성현|이재욱|최동민|홍인희|
:-:|:-:|:-:|:-:|:-:
<img src='https://user-images.githubusercontent.com/97818356/208237635-9bf65d96-ce27-4575-beb4-bb581a2c8e32.jpeg' height=80 width=80px></img>|<img src="https://user-images.githubusercontent.com/97818356/208237572-10eeebd0-9134-41ce-a9a2-a9577f8384e6.jpeg" height=80 width=80px/>|<img src='https://user-images.githubusercontent.com/108864803/208801820-5b050001-77ed-4714-acd2-3ad42c889ff2.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/108864803/208801708-fac3aade-f4bd-48c3-a623-973724ee68d0.png' height=80 width=80px></img>|<img src='https://user-images.githubusercontent.com/97818356/208237742-7901464c-c4fc-4066-8a85-1488d56e0cce.jpg' height=80 width=80px>|
[Github](https://github.com/datakim1201)|[Github](https://github.com/neulvo)|[Github](https://github.com/datakim1201)|[Github](https://github.com/unknownburphy)|[Github](https://github.com/inni-iii)


## :bowtie: Wrap up report
[project report 바로가기](https://github.com/boostcampaitech4lv23nlp1/level2_dataannotation_nlp-level2-nlp-07/blob/main/NLP%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%A0%9C%EC%9E%91_NLP_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(07%EC%A1%B0).pdf)  
&nbsp;  
## 프로젝트 개요

본 대회에서 제공받은 데이터는 대학 데이터셋입니다. 이번 대회에 더 몰입하기 위해 제공 받은 데이터와 추가로 크롤링한 데이터를 바탕으로  **<span style="color: #0000CD">대학에서 배울 수 있는 학문** 데이터 셋을 구축했습니다.

**자연어처리를 통해 대학 학문 관계 그래프를 만드는 이유**는 대학에서 배우는 여러 학문에 대한 구분이 명확하지 않다고 판단했기 때문입니다. 기존 분류에 대한 의견 및 이론 정립은 자주 나왔음에도 불구하고 많은 학생들이 여전히 명확한 구분을 하지 못하고 있습니다. 대량의 정보량을 토대로 자연어 학습을 하게 된다면 **<span style="color: #0000CD">‘실제로 쓰이는 바’를 토대로 지식그래프를 구축**할 수 있게 됩니다.

**대학에서 배울 수 있는 학문**들의 관계를 추출하고자 하는 연구자 분들에게 도움이 될 것이고 이외 **학문에 대한 지식 그래프 및 유관 분야 소개** 등으로의 확장 또한 가능할 것입니다.  
&nbsp;


## 프로젝트 절차 (22/12/5 ~ 22/12/15)
- 문장 추출 및 관계 brainstorming
- relation set 기반 pilot tagging
- relation map 및 가이드라인 작성
- main tagging
- 데이터 제작의 신뢰성 확인 IAA
    - IAA : 0.88
- 이전 대회 비교를 통한 성능 비교(f1-score)
    - baseline : 54.45
    - rbert : 83.417 
- 제작한 데이터를 통한 지식그래프 구축
    - networkx, pyvis를 통한 지식 그래프 구축  
&nbsp;  

### 지식 그래프 구축 예시
1. Type별 시각화    

<img src='https://user-images.githubusercontent.com/97818356/208237927-b61a1b3d-46eb-4883-982f-ac785b023073.png' height=300 width=300px> 

&nbsp;  
  
2. Word별 시각화  

<img src='https://user-images.githubusercontent.com/97818356/208237983-cecf5a84-2b3b-4de7-b4d9-e74365bb58c4.png' height=300 width=300px>&nbsp;&nbsp;<img src='https://user-images.githubusercontent.com/97818356/208238079-b65355c2-964f-4eb1-9c66-f1cf7b7bbee4.png' height=300 width=400px>

&nbsp;


```
for_Data Project/
│
├── calculate_IAA/ 
│   ├── calculate_IAA.ipynb
│   └── 수정후_relation.xlsx
│
│
├── notebook/ - for some_test
│   ├── json_pd.ipynb
│   └── network.ipynb
│
│
├── rbert/ - for further reading
│   ├── ...
│   └── model for some code check more detail..
│
│
├── src/ - for some visualization
│   ├── ...
│   └── some html ... png
│
│
├── config/ - abstract all config for each model
│   ├── config : for binary
│   ├── config_for_per
│   └── config_for_org
│
├── data/ 새로 추출한 데이터 
|
├── .gitignore
├── load_data.py
├── train.py
├── data_aug.py
├── README.md
│
│  
└── thanks for comming I'm Yeombora
```
