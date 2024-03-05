# import numpy as np
# import torch 
# # 두 개의 텐서 생성
# tensor1 = torch.zeros((128, 21, 18))  # [128, 21, 18] 모양의 영행렬 생성
# tensor2 = torch.ones((128, 1))        # [128, 1] 모양의 영행렬 생성
# tensor1_flatten = tensor1.flatten(start_dim=1)
# print(tensor1_flatten.shape)
# print(tensor2.shape)
# embedding = torch.nn.Embedding()
# tensor1_flatten_embedding = 

# # 두 번째 텐서를 첫 번째 텐서의 두 번째 축(인덱스 1)에 연결(concatenate)
# # axis=1로 지정하여 두 번째 축을 따라 연결합니다.
# result_tensor = torch.cat((tensor1_flatten,tensor2), dim=1)

# # 결과 확인
# print(result_tensor.shape)  # (128, 22, 19)

import torch
import torch.nn as nn

# 임베딩 레이어 초기화
# vocab_size = 10000  # 단어 집합의 크기
# embedding_dim = 300  # 임베딩 차원
# embedding = nn.Embedding(vocab_size, embedding_dim)

# # 임의의 입력 데이터 생성
# input_data = torch.LongTensor([[1, 2, 5, 4], [4, 3, 2, 9]])

# # 입력 데이터에 대한 임베딩 적용
# embedded_data = embedding(input_data)

# # 출력 확인
# # print(embedded_data)
# # print(embedded_data.shape)  # [2, 4, 300]
# # print(int(round(11/3)))

# tensor1 = torch.zeros((1,128,50))
# tensor2 = torch.zeros((1,128,1))

# print(torch.concat((tensor1,tensor2),dim=2).shape)

import torch
import torch.nn as nn

# 입력 데이터의 크기
input_size = 10

# 입력 데이터 생성 (예시)
input_data = torch.randn(1, 1, input_size)  # 배치 크기, 채널 수, 데이터 길이

# Convolutional Layer 정의
conv_layer = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)

# 패딩을 적용한 컨볼루션 연산 수행
output_data = conv_layer(input_data)

print("입력 데이터 크기:", input_data.shape)
print("출력 데이터 크기:", output_data.shape)
