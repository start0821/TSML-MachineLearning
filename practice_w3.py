import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import math
# np.set_printoptions(threshold=np.nan)

data = sio.loadmat('sentimentdataset.mat',chars_as_strings=1, matlab_compatible=1)

# print(data)
bagofword = data['bagofword']
# print(bagofword.size)
sentiment = data['sentiment'] # 문장이 부정적인지 긍정적인지에 대한 라벨링
sentiment = sentiment.astype(int)
words = data['word'] # 단어의 개수는 29717 ,backofword 하나당 array 원소도 동일한 개수이다.
                        # 알고보니 데이터가 좀 이상한거였다. 2115번째 word는 단어가 아니라 이상한 숫자들의 나열이다.
# print(words.shape)

word = []
for n in range(2000): # 단어를 2000개만 사용할 것이다. -> backofword array 원소도 2000개만 사용할것이다.
    word = word + [str(''.join(str(letter))) for letter in words[n][0]]
# print([str(''.join(letter)) for letter in words[0][0]]+[str(''.join(letter)) for letter in words[1][0]])
# print(words[2115][0])
# print(word[2115])

cell = 10 # training set 크기를 변화시키는 횟수
replication = 2 # 반복 횟수
numTesting = 50 # testing set 크기
NumWord = 2000

trainingAccuracy = np.zeros((replication,10))
# print(trainingAccuracy.shape)
testingAccuracy = np.zeros((replication,10))
avgTraining = np.zeros((cell,1))
stdTraining = np.zeros((cell,1))
avgTesting = np.zeros((cell,1))
stdTesting = np.zeros((cell,1))

for M in range(1,cell+1): # 트레이닝 셋 변화 횟수만큼 반복
    N = M * 10 # N : 트레이닝 셋 개수
    for rep in range(replication):
        sample = np.random.permutation(np.arange(198))

        # 데이터에 랜덤성 부여
        X = bagofword[sample]
        Y = sentiment[sample]
        # print(Y.shape)

        cntXbyY = np.ones((NumWord,2))/1000  # 0이되는것을 방지하기 위해 미리 좀 더해줌
        for i in range(NumWord):
            for j in range(N):
                if X[j,i] >= 1:
                    cntXbyY[i,Y[j]] = cntXbyY[i,Y[j]] + 1

        # cntY : Training set에 Positive sentiment와 Negative sentiment의 갯수
        cntY = np.zeros((2,1))
        for j in range(N): # Training set만큼 counting
            if Y[j] == 0:
                cntY[0] += 1
            else:
                cntY[1] += 1

        for i in range(NumWord):
            for j in range(2):
                if cntXbyY[i,j] > cntY[j]:
                    cntXbyY[i,j] = cntY[j]-1/1000
        # probsXbyY : Sentiment가 주어질 때, 각 단어가 해당 Sentiment를 가질 확률
        # 해당 단어가 등장하는 Positive(또는 Negative) Sentiment 문서의 갯수 / Positive(또는 Negative) Sentiment의 갯수
        probsXbyY = np.zeros((NumWord,2))
        for i in range(NumWord):
            for j in range(2):
                probsXbyY[i,j] = cntXbyY[i,j] / float(cntY[j])
                # if probsXbyY[i,j] < 1e-5:
                    # print("small")

        # probsY : 어떤 문서가 Positive 또는 Negative Sentiment를 가질 확률
        # Positive(또는 Negative) Sentiment인 문서 갯수 / 전체 문서 갯수
        probsY = np.zeros((2,1))
        for i in range(2):
            probsY[i] = cntY[i] / float(cntY[0]+cntY[1])

        # probsSentiment = np.zeros((198,2))
        logProbsSentiment = np.zeros((198,2));
        for i in range(198):
            for k in range(2):
                logProbsSentiment[i,k] = 0.;
                for j in range(NumWord):
                    if X[i,j] == 1:
                        logProbsSentiment[i,k] = logProbsSentiment[i,k] + math.log(probsXbyY[j,k]);
                    else:
                        # print(j,k)
                        # if probsXbyY[j,k] > 0.9:
                        #     print(probsXbyY[j,k])
                        #     print(cntXbyY[j,k])
                        #     print(float(cntY[k]))
                        logProbsSentiment[i,k] = logProbsSentiment[i,k] + math.log(1 - probsXbyY[j,k]);
                logProbsSentiment[i,k] = logProbsSentiment[i,k] + math.log(probsY[k]);

        # 각 문서에 대하여 Sentiment 값을 추정하여 결정
        estSentiment = np.zeros((198,1));
        for i in range(198):
            if logProbsSentiment[i,0] > logProbsSentiment[i,1]:
                estSentiment[i] = 0
            else:
                estSentiment[i] = 1

        cntCorrect = 0
        for i in range(N):
            if estSentiment[i] == Y[i]:
                cntCorrect += 1
        trainingAccuracy[rep,M-1] = cntCorrect/float(N);

        cntCorrect = 0
        for i in range(N,N+numTesting+1):
            if estSentiment[i] == Y[i]:
                cntCorrect +=1
        testingAccuracy[rep,M-1] = cntCorrect/float(numTesting)

# 정확도의 평균값 계산
    avgTraining[M-1] = np.mean(trainingAccuracy[:,M-1]);
    avgTesting[M-1] = np.mean(testingAccuracy[:,M-1]);
    # 정확도의 표준편차 계산
    stdTraining[M-1] = np.std(trainingAccuracy[:,M-1]);
    stdTesting[M-1] = np.std(testingAccuracy[:,M-1]);

plt.figure(1, figsize=(7,5))
plt.errorbar(np.dot(10,range(1,cell+1)),avgTraining,yerr = stdTraining/np.sqrt(replication), fmt='-o', color='r', label = "Training");
plt.errorbar(np.dot(10,range(1,cell+1)),avgTesting,yerr = stdTesting/np.sqrt(replication),  fmt='-o', color='b', label = "Testing");

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Number of Training Cases', fontsize = 14)
plt.ylabel('Classification Accuracy', fontsize = 14)

plt.show()
