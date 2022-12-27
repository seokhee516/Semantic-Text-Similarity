# 문장 간 유사도 측정
# 1. 프로젝트 개요

<aside>
    
- 부스트캠프 AI Tech에서 개최한 NLP 기초대회  
- 문장 간 유사도 측정: 의미 유사도 판별(Semantic Text Similarity, STS)이란 두 문장이 의미적으로 얼마나 유사한지를 수치화하는 NLP Task  
- 대회기간: 2022.10.26 ~ 2022.11.03
- 데이터셋: 학습 데이터셋 9,324개, 검증 데이터셋 550개, 평가 데이터는 1,100개. 평가 데이터의 50%는 Public 점수 계산에 활용되어 실시간 리더보드에 표기가 되고, 남은 50%는 Private 결과 계산에 활용되어 대회 종료 후 평가
- 평가방법: 0과 5사이의 유사도 점수를 예측. 피어슨 상관계수(Pearson Correlation Coefficient ,PCC) 지표

</aside>

### TimeLine

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/8c31dd14-8481-4733-8767-d13a6afa9076/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221201T083306Z&X-Amz-Expires=86400&X-Amz-Signature=be620327ae19cc6b55d440dd059d82d532f7a6795c4424638f16735a789e61a9&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

### 협업 방식

> **Notion**
> 

Team Notion에 각 팀의 현황과 실험 결과를 기록 및 공유.

![Untitled (33)](https://user-images.githubusercontent.com/86893209/207260426-837d1777-2d43-4842-a4d4-e9760fd75ed3.png)

> **Git**
> 

![Untitled (31)](https://user-images.githubusercontent.com/86893209/207260463-f6b316bc-2061-4057-8ecb-7aee0750ff9d.png)

![Untitled (32)](https://user-images.githubusercontent.com/86893209/207260493-8da5cdc9-3aec-4f1a-86b7-bc7a64312f4a.png)

master branch에서 baseline 수정 후, 팀원의 이름으로 분기를 나누어 작업 진행하였습니다. Wandb, config 파일 연결 등 분업하여 작업 후, 각자 branch로 merge하였습니다.

# 2. 프로젝트 팀 구성 및 역할

🔬**EDA** : 단익

> Exploratory Data Analysis, Reference searching
> 

🗂️**Data** : 재덕, 석희

> Data Augmentation, searching the pre-trained models
> 

🧬**MODEL** : 건우, 용찬

> to reconstruct the baseline, searching the pre-trained models
> 

# 3. 프로젝트 수행 절차 및 방법

![Untitled (41)](https://user-images.githubusercontent.com/86893209/209278066-02d413d5-7f67-4052-9309-0a767e68d349.png)

## 1) 탐색적 분석 및 전처리(EDA) - 학습 데이터 소개

![Untitled (40)](https://user-images.githubusercontent.com/86893209/209278059-825e3621-8016-4052-ba8c-2266c67c1252.png)

train.csv

- 두 문장 간의 유사도를 예측하는 것이 프로젝트의 최종 목표이고, 데이터셋은 train(9,324 rows)/dev(550 rows)/test(1,100 rows) 비율로 나누어, csv형태로 제공되었습니다.
- 각 문장의 출처는 국민청원 게시판 제목, 네이버 영화 감성 분석 코퍼스, 업스테이지 슬랙 데이터이며. 각 데이터별 유사도(Label) 점수는 여러명의 사람이 공통의 점수 기준으로 두 문장간의 점수를 평균낸 값입니다.

![Untitled (35)](https://user-images.githubusercontent.com/86893209/207260557-0ad7c27d-c969-45da-8ced-07ed945a81ba.png)

train.csv : 9,324 rows

![Untitled (36)](https://user-images.githubusercontent.com/86893209/207260585-e63f91b9-1cb7-4399-9e24-085801d3d11c.png)

dev.csv : 550 rows

- train 데이터셋의 Label별 데이터 분포 시각화를 통해, Label 0으로 쏠린 불균형 데이터를 관측했습니다. 반면, dev.csv 데이터는 모든 label의 분포가 대체로 균일한 편으로 관측되었습니다.
- train의 데이터 불균형을 해소하기 위해, ****label 0인 데이터를 줄여서 다른 label과 분포를 맞추거나, label 5를 늘려서 균형을 맞추는 방법으로 데이터 클래스 불균형을 해결하고자 했습니다.(Data Augmentation)

## 2) Modeling

### Baseline Code 수정

- Wandb, Wandb Sweep 구현
- yaml+OmegaConf+Shell 활용한 모델학습 및 실험관리 편의성 증대

### 가장 좋은 Pre-trained Model 선택

- 한국어 기반 RoBERTa, ELECTRA Pre-trained Model 들을 비교해보았고, snunlp/KR-ELECTRA-discriminator가 가장 좋은 성능을 보였습니다.
- 이후 snunlp/KR-ELECTRA-discriminator 모델을 기반으로 Data Augmentation 최적화를 진행하였습니다.

## 3) Data pre-processing

### Data Augmentation

Baseline Model(klue/roberta-base, Loss: L1, Optimizer: AdamW) 기준으로 아래 4가지 증강기법 및 스무딩 기법을 적용하여 Data pre-processing을 진행하였습니다. 원본 데이터와 증강된 데이터의 비율을 조정 하면서 결과를 확인하였고, 여러 증강 기법을 중복 적용하는 등 최적의 조합을 찾아내고자 하였습니다. 그 외, 학습의 속도를 위해 learning rate를 늘려주거나, general한 모델을 만들기 위해 batch size등을 늘려주었습니다.

- **Back Translation¹⁾**
    - 한국어에서 영어 번역 후, 영어에서 한국어 역번역
    - 역 번역 시 부적절한 번역 결과와 발생하여 일관된 점수 기준이 중요한 STS Task에 적절하지 못한 기법이라 판단하여 제외하였습니다.
- **Copied Translation²⁾**
    - sentence1을 sentence 2로 복사하여, label 5 데이터 생성
    - Train Dataset 분포 분석 결과 5 Label 데이터가 전체 데이터셋의 1%이기에 Sentence가 서로 같은 문장을 원 데이터셋에서 샘플링하여 5 Label 데이터를 추가하였습니다.
- **Swap** **Sentence**
    - sentence1과 sentence 2의 순서를 바꿔줌
    - Sentence 1과 Sentece 2의 Segment Embedding 값이 다르기에 변경 시 유의미한 데이터 증강이 될 것이라고 분석하였습니다. 시도한 방법 중 가장 효과가 좋았습니다.
- **Reverse Text³⁾**
    - 문자를 역순으로 생성
    - 단독 사용시 효과가 있었고 이를 통해 유의미한 노이즈 값을 생성할 수 있을 것이라 분석했지만 여러 기법과 함께 사용시 성능이 하락하여 제외하였습니다.
- **Label Smoothing**
    - label 0 데이터 제거
    - Train Dataset의 50% 이상이 0 Label 이기에 해당 Label을 50% 언더 샘플링 하였습니다.
    - 이를 Copied Translation과 함께 사용할시 효과가 좋았습니다. 이를 원 분포인 Positive Skewness 분포에서 비교적 Uniform한 분포로 변경된 결과라고 분석하였습니다.

실험 결과, learning rate 1e-5, batch size 16에서는 **Copied Translation, Reverse Text**, learning rate 2e-5, batch size 32에서는 **Swap Sentence, Label Smoothing, Copied Translation**이 유의미한 성능 향상을 보였습니다.

## 4) O**ptimization**

### Hyperparameter 실험 및 비교

- Loss, Batch Size, Learning rate, Data에 따른 실험 및 비교
    
    
    | Loss | MSE | L1 |  |
    | --- | --- | --- | --- |
    | Batch Size | 16 | 32 |  |
    | Learning rate | 1e-5 | 3e-5 | 5e-5 |
    | Data | Label Smoothing 0, 
    Copied Translation Label 5 | Swap Sentence |  |
    
    | Model | Loss | Learning rate | Batch Size | Val Pearson |
    | --- | --- | --- | --- | --- |
    | RoBERTa Large - Label Smoothing 0, Copied Translation Label 5 | MSE | 7e-6 | 8 | 0.9256 |
    | ELECTRA - Swap Sentence | MSE | 3e-5 | 32 | 0.9287 |
    | ELECTRA - Label Smoothing 0, Copied Translation Label 5, Swap Sentence | MSE | 3e-5 | 16 | 0.9309 |

### snunlp/KR-ELECTRA-discriminator 최적화

- 가장 성능이 좋았던 Pre-trained 모델인 snunlp/KR-ELECTRA-discriminator와 Data Augmentation 실험 결과를 기반으로 최적 조합을 실험하였습니다.
- 실험 결과, **Swap Sentence**가 유의미한 성능 향상을 보였습니다. 또한 베이스라인에 사용되었던 L1 Loss 보다 **MSE Loss**가 더 높은 성능을 보여, 이후 실험에서는 **MSE Loss**를 사용하였습니다.

### RoBERTa Large 최적화

![Untitled (37)](https://user-images.githubusercontent.com/86893209/207260672-5d7305d9-1ab6-4b69-addf-1ddc5e709eb4.png)

- klue/roberta-large의 경우 모델의 크기가 커서 학습이 수행되지 않는 문제가 발생하여, **batch size와 learning rate를 조정**하여 최적화하였습니다. Data Augmentation 결과를 기반으로, 가장 유의미 했던 Swap Sentence, Label Smoothing 0 및 Copied Translation Label 5, Reverse Text 20% 데이터를 각각 실험하였습니다.
- 실험 결과, **Label Smoothing 0 및 Copied Translation Label 5** 데이터의 **MSE Loss**, **Learning Rate 7e-6, Batch Size 8** 조건에서 **Val Pearson 0.9256**으로 가장 높은 성능을 확인하였습니다.

### ELECTRA 최적화

![Untitled (38)](https://user-images.githubusercontent.com/86893209/207260693-86b3c11c-852f-4000-9034-30788a94cf17.png)

- 한국어 ELECTRA 모델 3개(monologg/koelectra-base-v3-discriminator, beomi/KcELECTRA-base, snunlp/KR-ELECTRA-discriminator)와 데이터 증강에서 유의미한 성능향상을 보인 Swap Sentence, Label Smoothing 0, Copied Translation Label 5 데이터를 기준으로, Learning Rate, Batch Size 최적화를 진행하였습니다.
- 실험 결과, **snunlp/KR-ELECTRA-discriminator** 모델, **Label Smoothing 0, Copied Translation Label 5, Swap Sentence** 데이터의 **Learning Rate 3e-5 Batch Size 16** 조건에서 **Val Pearson 0.9309**으로 **단일 모델 중 가장 높은 성능**을 확인하였습니다.

## 5) Ensemble

- 평가 지표인 Pearson의 경우 선형이기에 Outlier에 취약한 특성이 있음. 이를 해결하기 위해 가중 평균을 도입, Outlier의 영향력을 줄였습니다.⁴⁾
- 앙상블은 소프트보팅 방식을 채용하여 각 모델의 결과를 더해주고 평균을 내주는 대신 각 모델의 성능을 가중치로 두어서 가중 평균을 구하는 방식으로 구현했습니다. 각 모델의 성능을 softmax 층에 통과시켜서 확률로 변환한 후 각 모델이 출력한 logit 값과 곱해주어서 전부 더해주었습니다.
- Swap Sentence 기법을 적용하여  Positive Skewness 분포인 데이터와 Copied Translation과 Label Smoothing을 적용하여 Uniform 분포를 가진 데이터를 학습한 모델을 앙상블하여 Test Dataset 분포에 의존적이지 않으며 General한 모델을 설계하였습니다.
- 처음에는 제일 성능이 좋은 klue/roberta-large와 snunlp/KR-ELECTRA-discriminator 1개씩 가져와서 성능을 91.24에서 92.25로 개선했습니다. 그 후 각 모델을 3개씩 앙삼블한 모델로 성능을 92.69까지 올렸고, 모델 간의 낮은 상관관계를 갖고 있으면 앙상블이 효과적이라는 근거를 토대로 다양한 모델을 앙상블한 결과 최고 성능인 92.9가 나왔습니다
    
    
    | klue/roberta-large 모델 최고 성능 1개 | snunlp/KR-ELECTRA-discriminator 최고 성능 1개 | 성능 개선: | 0.9124 → 0.9225 |  |  |
    | --- | --- | --- | --- | --- | --- |
    | klue/roberta-large 모델 3개 | snunlp/KR-ELECTRA-discriminator 3개 | 성능 개선:  | 0.9225 → 0.9269 |  |  |
    | klue/roberta-large 모델 3개 | snunlp/KR-ELECTRA-discriminator 3개 | beomi/KcELECTRA-base 1개 | • monologg/koelectra-base-v3-discriminator 1개 | 성능 개선: | 0.9269 → 0.9290 |

# 4. 프로젝트 수행 결과

![Untitled](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5bf59d2a-8e26-4bf8-9ad6-217b2506cffe/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221201%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221201T083349Z&X-Amz-Expires=86400&X-Amz-Signature=bad420e9e339a79dd06db5ee818f494af7ebdee4bad66c5908e8ca04fc1ff245&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

- 최종 pearson : 0.9368
- 14팀 중 public 4위 private 3위

# 5. 결론

### 데이터의 특성을 잘 반영하는 기초 모델 선정 이후 다양한 실험을 통해 최적화 및 앙상블 수행

- 데이터 분석을 통한 데이터 품질 개선(oversampling, data augmentation)
- 데이터셋에 적합한 Pretrained Model 선정 및 최적화
- 다양한 결과에 대한 앙상블(Soft Voting)을 수행

# 6. Appendix

- Pre-trained Model 선택
    
    
    | Model | Epoch (Earlystopping/Max epoch, Best Check point) | Val loss | Val Pearson |
    | --- | --- | --- | --- |
    | klue/roberta-small | 4/100, 4 | 0.6108 | 0.8523 |
    | klue/roberta-base | 4/100, 4 | 0.533 | 0.8916 |
    | jhgan/ko-sroberta-multitask | 3/100, 3 | 0.5149 | 0.8828 |
    | beomi/KcELECTRA-base | 8/20, 4 | 0.4385 | 0.9113 |
    | snunlp/KR-ELECTRA-discriminator | 9/15, 7 | 0.4705 | 0.9242 |
- Data Augmentation
    
    
    | Model | Epoch | Learning rate | Batch Size | Data Augmentation | Val loss | Val Pearson  |
    | --- | --- | --- | --- | --- | --- | --- |
    | klue/roberta-base | 4 | 1e-5 | 16 | Baseline | 0.533 | 0.8916 |
    |  | 9 | 1e-5 | 16 | 원본:Back Translation 50% (2:1) | 0.5864 | 0.8655 |
    |  | 8 | 1e-5 | 16 | 원본:Back Translation 33% (3:1) | 0.4987 | 0.8958 |
    |  | 16 | 1e-5 | 16 | 원본:Back Translation 25% (4:1) | 0.4308 | 0.91 |
    |  | 16 | 1e-5 | 16 | Copied Translation Label 5 50% | 0.4707 | 0.9126 |
    |  | 5 | 1e-5 | 16 | Copied Translation Label 5 20% | 0.5326 | 0.9024 |
    |  | 10 | 1e-5 | 16 | Copied Translation Label 5 10% | 0.5083 | 0.9082 |
    |  | 5 | 1e-5 | 16 | Reverse Text 50% | 0.4957 | 0.8961 |
    |  | 14 | 1e-5 | 16 | Reverse Text 20% | 0.4464 | 0.9169 |
    |  | 19 | 1e-5 | 16 | Reverse Text 10% | 0.4869 | 0.9074 |
    |  | 3 | 1e-5 | 16 | 원본:Exchange Sentence : Reverse Text 10% (1:1:0.2) | 0.4695 | 0.908 |
    |  | 4 | 1e-5 | 16 | 원본:Exchange Sentence : Reverse Text 20% (1:1:0.4) | 0.4384 | 0.9118 |
    | klue/roberta-base | 4 | 2e-5 | 32 | Baseline | 0.5919 | 0.8616 |
    |  |  | 2e-5 | 32 | Swap Sentence | 0.5013 | 0.8967 |
    |  |  | 2e-5 | 32 | Swap Sentence : Back Translation (2:1) | 0.5008 | 0.8922 |
    |  |  | 2e-5 | 32 | Swap Sentence : Back Translation (1:1) | 0.4978 | 0.8845 |
    |  |  | 2e-5 | 32 | Swap Sentence, Label Smoothing 0 50% | 0.4892 | 0.8986 |
    |  |  | 2e-5 | 32 | Swap Sentence, Label Smoothing 0 25% | 0.4722 | 0.8963 |
    |  |  | 2e-5 | 32 | Swap Sentence, Label Smoothing 0 50%, Copied Translation Label 5 | 0.4801 | 0.8931 |
    |  |  | 2e-5 | 32 | Swap Sentence, Label Smoothing 0 25%, Copied Translation Label 5 | 0.4536 | 0.9123 |
- snunlp/KR-ELECTRA-discriminator 최적화
    
    
    | Model | Epoch | Loss | Data Augmentation | Val loss | Val Pearson |
    | --- | --- | --- | --- | --- | --- |
    | snunlp/KR-ELECTRA-discriminator | 10 | MSE | Swap Sentence  | 0.3914 | 0.9238 |
    |  | 6 | MSE | Swap Sentence, Label Smoothing 0 50% | 0.4252 | 0.9096 |
    |  | 12 | L1 | Swap Sentence   | 0.5068 | 0.9001 |
    |  | 17 | L1 | Swap Sentence, Label Smoothing 0 50% | 0.4605 | 0.9172 |
    |  | 12 | L1 | Swap Sentence, Label Smoothing 0 50% + Copied Translation Label 5 50% | 0.4484 | 0.9235 |
    |  | 10 | L1 | Swap Sentence, Label Smoothing 0 50%, Reverse Text 20% | 0.4486 | 0.919 |
- RoBERTa Large 최적화
    
    
    | Model | Epoch  | Learning rate | Batch Size | Data Augmentation | Val loss | Val Pearson |
    | --- | --- | --- | --- | --- | --- | --- |
    | klue/roberta-large | 9 | 1e-6 | 8 | Swap Sentence | 0.4043 | 0.913 |
    |  | 5 | 3e-6 |  | Swap Sentence | 0.4513 | 0.9116 |
    |  | 11 | 5e-6 |  | Swap Sentence | 0.354 | 0.9208 |
    |  | 7 | 7e-6 |  | Swap Sentence | 0.379 | 0.9201 |
    |  | X | 1e-6 |  | Label Smoothing 0, Copied Translation Label 5 | X | X |
    |  | 10 | 3e-6 |  | Label Smoothing 0, Copied Translation Label 5 | 0.3726 | 0.9171 |
    |  | 6 | 5e-6 |  | Label Smoothing 0, Copied Translation Label 5 | 0.3845 | 0.9121 |
    |  | 8 | 7e-6 |  | Label Smoothing 0, Copied Translation Label 5 | 0.3363 | 0.9256 |
    |  | 5 | 5e-6 |  | Copied Translation Label 5 | 0.4841 | 0.9019 |
    |  | 8 | 5e-6 |  | Reverse Text 20% | 0.4631 | 0.91 |

1.  [Data Augmentation using Back-translation for Context-aware Neural Machine Translation](https://aclanthology.org/D19-6504.pdf)

2.  [신경망 기계번역에서 최적화된 데이터 증강기법 고찰](https://koreascience.kr/article/CFKO201930060755841.pdf) - “실험결과 Back Translation과 Copied Translation을 함께 적용하여, 4대3의 상대적 비율을 적용하여 학습을 진행했을 때 가장 높은 BLEU 점수를 보였다.”

3.  [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf) - “… reversing the order of the words in all source sentences (but not target sentences) improved the LSTM’s performance markedly”

4. [Pearson Coefficient of Correlation Explained](https://towardsdatascience.com/pearson-coefficient-of-correlation-explained-369991d93404) - “… Pearson’s correlation coefficient, r, is very sensitive to outliers, which can have a very large effect on the line of best fit and the Pearson correlation coefficient.”
