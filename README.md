## Data annotation


## 본 대회 설명  
본 대회에서 제공받은 데이터는 대학 데이터셋입니다. 이번 대회에 더 몰입하기 위해 제공 받은 데이터와 추가로 크롤링한 데이터를 바탕으로  **대학에서 배울 수 있는 학문** 데이터 셋을 구축했습니다.

**자연어처리를 통해 대학 학문 관계 그래프를 만드는 이유**는 대학에서 배우는 여러 학문에 대한 구분이 명확하지 않다고 판단했기 때문입니다. 기존 분류에 대한 의견 및 이론 정립은 자주 나왔음에도 불구하고 많은 학생들이 여전히 명확한 구분을 하지 못하고 있습니다. 대량의 정보량을 토대로 자연어 학습을 하게 된다면 **‘실제로 쓰이는 바’를 토대로 지식그래프를 구축**할 수 있게 됩니다.

**대학에서 배울 수 있는 학문**들의 관계를 추출하고자 하는 연구자 분들에게 도움이 될 것이고 이외 **학문에 대한 지식 그래프 및 유관 분야 소개** 등으로의 확장 또한 가능할 것입니다.

## 우리의 목적 
- 데이터 제작의 신뢰성 확인 IAA
- 이전 대회 비교를 통한 성능 비교 
    - baseline : for check base
    - rbert : for further reading 
- 제작한 데이터를 통한 지식그래프 구축

    - networkx, pyvis를 통해 지식 그래프 구축
```
for_binary/
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