### cs231n study, Assign#2

### Dropout, Drop Connect / Batch Normalization / Mini batch

#### 18.10.09

#### Dropout

학습시 overfitting을 방지하기 위한 Skill.
일부 Neuron을 deactivate 시킨다.
우리가 training을 시키면서 변화하는 것은 data(input)가 아니라 weight(가중치) 이다.
forward pass 시 일부 neuron을 deactivate 시키게 되면 overfit을 줄일 수 있는 효과가 존재한다.

##### ![img](https://t1.daumcdn.net/cfile/tistory/2237423D57A0299F2A)



why?
그 전에 overfit을 방지하는 고전적인 세가지 방법을 보자

1. overfit을 하지 않을 정도의 많은 양의 data를 모아 학습시킨다.
   overfitting은 보통 적은 양의 data를 사용할 때 발생하므로, 처음부터 data의 양을 많게 한다.
2. feature의 갯수를 줄인다. 
   너무 많은 feature는 오히려 overfit을 유발한다.
   예를 들어 손가락이 여섯개인 사람 6명을 넣어 학습 데이터로 삼아 학습을 한다고 하자
   이 학습에서는 사람의 특징이 손가락 6개가 되어버린다.
   너무 specific한 feature는 overfit을 유발 할 수 있다.
3. regularization 
   weight값이 너무 크게 변하지 않게 규제를 해준다.

이중에서 dropout 방법은 2번의 방법에 해당된다고 할 수 있다.

##### ![img](https://t1.daumcdn.net/cfile/tistory/217D274F57A0248E20)



### Drop Connect

Dropout의 일반화된 형태이다.
dropout는 퍼셉트론(뉴런)을 끊어 앞뒤로 연결된 가중치를 없애는 방법이지만
(뉴런의 출력을 없앤다)
drop conncet는 가중치만 없앤다.
즉, epoch마다 (매 입력마다) 뉴런의 입력을 죽인다. (무슨말인지 이해가 잘안간다.)
dropout은 node 자체를 deactivate 하지만 
dropconnect는 node를 살린채로 weight를 0으로 설정해준다고 이해하면 될 듯하다.

#####![áá³áá¦ááµ](https://taran0sh.files.wordpress.com/2018/04/e18489e185b3e1848fe185a6e1848ee185b529-e1524148957328.png?w=620)



#### Mini batch

전체 학습 데이터를 나누어서 사용한다.
http://dambaekday.tistory.com/1 참조.
데이터를 한개 쓰면 빠르지만 너무 헤매고, 전체를 쓰면 정확하지만 너무 느립니다. 즉 적당히 빠르고 적당히 정확한 길을 찾기 위해 사용한다.

#### Batch Normalization

Batch(data)를 normalization 시켜서 (평균0, 표준편차 1)
네트워크 내부의 데이터를 안정화 시켜 학습자체를 안정적으로 할 수 있게한다.
좀 더 정확하게 말하면 Gradient Vanishing / Gradient Exploding 문제를 해결하기 위해 나온 기법이다.
이전에는 Activation 함수의 변화 (ReLU 등), Careful Initialization, small learning rate 을 통하여 
간접적으로 데이터를 안정화 시켰지만, 해당 기법을 사용하여 직접적으로 data를 조작, 안정화 시킨다.

#####![normalize](https://shuuki4.files.wordpress.com/2016/01/bn1.png)

데이터를 중앙으로 모아준다고 생각하면 될 것 같다. (scale의 제약도 함께)
표준값 z를 구해준다 생각하면 됨
즉 모든 데이터를 일정 범위에 국한시켜 보는것이다
여기서 scale, shift는 training되는 parameter이다.

Batch Normalization의 아이디어는 Internal Covariance Shift라는 data의 불안정화를
근본적으로 막고자 탄생하였다.
**Internal Covariance Shift** 는 Network의 각 층이나 Activation 마다 
input의 distribution이 달라지는 현상을 의미한다
(최초 input이 각 층을 통해 전달되면서 값이 조금씩 바뀌는데 이를 해결해주기 위한 것인듯)
이를 막기위해서 각 층의 data를 넣기전 (평균 = 0, 표준편차 = 1) normalization을 시켜주어
distribution이 달라지는 것을 막을 수 있다. 이러한 과정을 **whitening** 이라 한다. (이전에 쓰임)
이는 input의 feature들을 uncorrelated 하게 만들어주고, 각각의 variance를 1로 만들어주는 작업이다.
(normalize 시키고, data의 feature를 서로 영향이 없게 만들어 주는 작업)
하지만 해당 작업은 **계산량이 많아지고** (feature들의 영향을 보기위해 covariance matrix 계산 필요)
**일부 parameter는 무시가 된다.**
(예를 들어 input u를 받아 x=u+b라는 output을 내놓고 적절한 bias b를 학습하려는 네트워크에서 x에 E[x]를 빼주는 작업을 한다고 생각해보자. 그럴 경우 E[x]를 빼는 과정에서 b의 값이 같이 빠지고, 결국 output에서 b의 영향은 없어지고 만다. E[x] = ${1\over n}\sum{u+b}​$

이를 해결하기 위하여 Batch Normalization 이 등장한다.
**Batch Normalization의 핵심 Idea**는 다음과 같다.

- 각각의 feature들이 이미 uncorrelated 되어있다고 가정하고, feature 각각에 대해서만 scalar 형태로 mean과 variance를 구하고 각각 normalize 한다.
  (계산량 문제 해결)
- 단순히 mean과 variance를 0, 1로 고정시키는 것은 오히려 Activation function의 nonlinearity를 없앨 수 있다. 예를 들어 sigmoid activation의 입력이 평균 0, 분산 1이라면 출력 부분은 곡선보다는 직선 형태에 가까울 것이다. 또한, feature가 uncorrelated 되어있다는 가정에 의해 네트워크가 표현할 수 있는 것이 제한될 수 있다. 이 점들을 보완하기 위해, normalize된 값들에 scale factor (gamma)와 shift factor (beta)를 더해주고 이 변수들을 back-prop 과정에서 같이 train 시켜준다.
  (평균 0, 분산 1로 부터 오는 문제를 해결)
- training data 전체에 대해 mean과 variance를 구하는 것이 아니라, mini-batch 단위로 접근하여 계산한다. 현재 택한 mini-batch 안에서만 mean과 variance를 구해서, 이 값을 이용해서 normalize 한다.
  (data는 mini batch단위로 잘게 쪼개어)



Batch Normalization은 다음과 같은 장점이 존재

1. 기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다. 이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다. 따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.
   (학습속도가 빨라진다)
2. Batch Normalization의 경우 자체적인 regularization 효과가 있다. 이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며, 나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.) . Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다.
   (Dropout 사용하지 않음)



참고문헌

* http://dongyukang.com/%EB%B0%B0%EC%B9%98-%EC%A0%95%EA%B7%9C%ED%99%94-%EB%85%BC%EB%AC%B8%EC%9D%84-%EC%9D%BD%EA%B3%A0/
* https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/
* https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220808903260&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F
* https://de-novo.org/tag/pca/ dropconnect