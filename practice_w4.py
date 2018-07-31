import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat('sentimentdataset.mat', chars_as_strings=1, matlab_compatible=1)

bagofword = data['bagofword'];
sentiment = data['sentiment'];
sentiment = sentiment.astype(int);
words = data['word'];

word = []
for n in range(2000):
    word = word + [str(''.join(letter)) for letter in words[n][0]]


cell = 10;  # Training set의 크기를 변화시키는 횟수# Train
replication = 5; # 반복횟수
numTesting = 50; # Testing set의 크기
numWord = 100; # 데이터의 크기 - 모델을 만들 때 사용하는 단어의 개수

trainingAccuracy = np.zeros((replication, 10));
testingAccuracy = np.zeros((replication, 10));
avgTraining = np.zeros((cell, 1));
stdTraining = np.zeros((cell, 1));
avgTesting = np.zeros((cell, 1));
stdTesting = np.zeros((cell, 1));

# Gradient Ascent의 상수
h = 0.5; # theta가 업데이트 되는 속력
threshold = 0.1; # 업데이트 되는 값이 작은 경우, theta가 local maximum에 수렴할 수 있으므로 threshold를 설정함
for M in range(1,cell+1):
    N = M*10
    for rep in range(replication):
        sample = np.random.permutation(np.arange(198))

        numFeatures = numWord + 1
        X = np.ones((np.size(sample),numFeatures))
        tempMat = bagofword[sample]
        tempMat = tempMat[:,range(numWord)]
        X[:,range(1,numFeatures)] = tempMat
        Y = sentiment[sample]

        theta = np.ones((numFeatures,1))
        itr = 500

        cntItr = 0
        for k in range(itr):
            thetanew = np.zeros((numFeatures,1))
            for i in range(numFeatures):
                temp = 0;
                for j in range(N):
                    Xtheta = 0;
                    Xtheta = np.dot(X[j,:],theta)[0]
                    temp += X[j,i] * (Y[j][0] - np.exp(Xtheta)/(1+np.exp(Xtheta)))
                temp = temp * h
                thetanew[i] = theta[i] + temp
            diff = np.sum(np.abs(theta-thetanew))
            if diff/(np.sum(np.abs(theta))) <threshold:
                break;
            cntItr += 1
            theta = thetanew

        # probsSentiment : 각 문서가 positive 또는 negative sentiment를 가질 확률
        probsSentiment = np.zeros((198,2))
        for i in range(198):
            Xtheta = np.dot(X[i,:],theta)[0]
            probsSentiment[i,0] = 1/(1+np.exp(Xtheta))
            probsSentiment[i,1] = 1-probsSentiment[i,0]

        # MCLE를 이용해서 각 문서의 sentiment를 추정함
        estSentiment = np.zeros((198,1))
        for i in range(198):
            if probsSentiment[i,0] > probsSentiment[i,1]:
                estSentiment[i] = 0
            else:
                estSentiment[i] = 1

        cntCorrect = 0;
        for i in range(N):
            if estSentiment[i] == Y[i]:
                cntCorrect += 1
        trainingAccuracy[rep,M-1] = cntCorrect / float(N)

        cntCorrect = 0;
        for i in range(N,N+numTesting): # 모든 testing set에 대하여 추정값과 실제값을 비교함
            if estSentiment[i] == Y[i]:
                cntCorrect = cntCorrect + 1;
        testingAccuracy[rep,M-1] = cntCorrect / float(numTesting); #반복횟수에 따른 training set 크기별 testing accuracy를 저장함

    # replication된 정확도의 평균값 계산
    avgTraining[M-1] = np.mean(trainingAccuracy[:,M-1]);
    avgTesting[M-1] = np.mean(testingAccuracy[:,M-1]);

    # replication된 정확도의 표준편차 계산
    stdTraining[M-1] = np.std(trainingAccuracy[:,M-1]);
    stdTesting[M-1] = np.std(testingAccuracy[:, M-1]);


plt.figure(1, figsize = (7,5));
plt.errorbar(np.dot(10,range(1,cell+1)), avgTraining, yerr = stdTraining/np.sqrt(replication), fmt='-o', color='r', label="Training");
plt.errorbar(np.dot(10,range(1,cell+1)), avgTesting, yerr = stdTesting/np.sqrt(replication), fmt='-o', label="Testing");

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Number of Training Cases', fontsize = 14)
plt.ylabel('Classification Accuracy', fontsize = 14)

plt.show();
