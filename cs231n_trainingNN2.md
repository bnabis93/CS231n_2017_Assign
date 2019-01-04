# Training NN2, cs231n

* ### Optimization

* ### Regularization

* ### Transfer learning

---

#### Optimization

lost function으로써 나의 model을 평가 할 수 있다.
여튼 가장 lowest value를 찾아야 한다. (우리의 목표)
direction이 gradient가 감소하는 방향으로 잡히고, minima를 찾게된다면
정말 이상적이겠지만 현실을 생각보다 녹록치 않다.

##### SGD(Stochastic Gradient Descent)

확률적 경사 하강법이라고 직역을 할 수 있다.
해당 방법은 Gradient를 계산 할 때, mini - batch에서의 데이터에 대해 loss를 계산한다.
$\theta  = \theta - \eta \nabla Loss$ , $\eta$ 는 step size(미리 정해둔 값)
즉, 모든 경우를 보는게 아닌 특정 구역의 값만을 봄으로 써, global한 방법보다 속도를 올린다.
하지만 이 방법의 경우 문제점이 존재한다. (성능이 안좋다.)
Gradient 방향이란것이 항상 minima를 향한다는 보장이 존재하지 않는다. (local minima)
그리고 dimension이 높아질수록 성능이 안좋아진다.
문제점은 정리하자면 다음과 같다.

1. Speed. (very low)
2. Saddle point / local minima problem
3. problem from mini-batch (noisy data

따라서 이러한 문제를 해결하기 위하여 Momentum이 등장하였다.

##### Momentum

SGD값에 일종의 관성을 주는 것 => Saddle point / local minima 잘 벗어나겠지!
Velocity값을 추가한다. 
이전의 SGD는 step size를 곱해주는 반면, momentum의 경우 속도값을 더함으로 써 
local mimima, saddle point와 같은 loss가 0이 되는 지점에서도 관성을 줌으로 써 
더 좋은 결과를 얻을 수 있다.
하지만 minima 근처에서 좋지않을 결과를 낼 것 같다. (왔다갔다왔다갔다)
그리고 변수를 별도로 저장해야 하므로, 메모리 효율에서 좋지 않아진다.

##### Nesterov Gradient (momentum)

#####![Difference between Momentum and NAG. Picture from CS231.](http://cs231n.github.io/assets/nn3/nesterov.jpeg)

기존의 momentum과 조금 다른방식으로 momentum을 update한다.
그림을 참조하면 이해하기 쉽다는데… 흠?̊̈
이전 momentum에서는 velocity값을 구하고, 그 후, gradient를 update해준다.
Nesterov에서는 velcity update시, loss를 update해주는데, 이 때, velocity값도 넣어 loss를 update해준다.
momentum의 문제점을 좀 더 개선 한 것으로 이해하고 키워드만 기억해놓자.

##### AdaGrad

각 변수들을 고려한 optimization 방법이다.
main idea는 다음과 같다. 
많이 등장한 변수들은 이전에 이미 update가 많이 되었을 것 이므로, 적은 step size로 이동하고
적게 등장한 변수들은 더 큰 step size로 이동하자.
$G_{t} = G_{t-1} + (\nabla Loss(\theta_{t}))^2$

$\theta_{t+1} = \theta_{t} - {\eta \over \sqrt{G_{t} + \epsilon}} * \nabla Loss(\theta_{t}) $

step size가 계속해서 줄어든다는 문제가 발생.
이러한 문제점을 해결하고자 나온것이 RMSProp

##### RMSProp

$G_{t} = \gamma G_{t-1} +(1-\gamma) (\nabla Loss(\theta_{t}))^2$

즉, 제곱을 평균으로 바꿔줌. => Gt가 무한정 커지지 않으며 변수간 상대적 차이는 유지가 가능하다. 
즉, AdaGrad의 문제점인 Step size가 계속해서 줄어든다는 것을 해결하였고,
AdaGrad의 main idea는 계속해서 유지를 하고 있다.

##### Adam

RMSProp + momentum 
(초반 momentum이 overshoot 이점, 후반은 RMS 이점 취하지 않았을까 예)
First moment / second moment로 equation이 진행이 된다.
완벽해보이는 Adam에도 문제점이 존재한다.
최초 단계에서 Adam은 문제점이 있다.
init단계에서 second moment를 0로 초기화 할 것이다. 그리고 update를 해나갈 것 이다.
이때 한번에 엄청 멀리 나갈 수 도 있다. (식을 봐야겠다)
Bias correction 이라는 step 이 추가된다. 

모든 Optimizer는 최초 hyperparameterd인 learning rate를 잘 정해주는것이 중요하다.

first order optimization / second order optimization 존재
second order는 아직까지 연산량이 너무나 많다.
=> estimation 해서 사용 (Taylor) 그리고 learning rate가 존재하지 않음
(BGFS / L-BFGS) => Hessian 사용

----



#### Model ensemble

그래. optimizer도 좋고 model 도 양호해. 하지만 train과 val간의 gap이 여전히 존재
Train과 val의 Accuracy gap을 줄이기 위함 => model ensemble
다양한 모델의 결과를 결합한다 (ensemble 한다!)
model 을 ensemble하기 위한 조건은

* 각 model 은 indépendant 해야 함

즉, model을 여러개 합침으로써 더 좋은 결과를 얻게하는 방법이라 할 수 있다.
(hyperparameter가 같을 필요는 없다.)

그렇다면 single model에서는 어떻게 성능을 높일 수 ㅣㅇㅆ을까?

----



#### Regularization

내가 이전부터 이해를 하지 못했던 개념이다. 
직관적으로는 이해가 가는데, 수식적으로는 이해가 잘 가지 않는다.
직관적으로는 Loss를 구할 때 일종의 패널티를 줌으로 써 구하는 방법이라 할 수 있다.
Weight값이 커지게 되는 경우 우리가 보고자 하는 범위에서 벗어난 것 까지 보게 될 수도 있다.
따라서 적절한 weight값을 유지하는 것은 중요한데
이를 위해 regularization을 하여 weight값이 너무 높아지는 것을 방지하는 것 이다.

##### L1 / L2 regularization

##### Dropout

dropout도 regularization 전략중 하나였군… 허허...
dropout mask는 randomly하게 input 에 대해 적용이 된다.
하지만 test time에서 dropout mask의 randomness는 오히려 독이 된다.
예를 든것이 페이스북에서 이미지 분류를 한다고 하자.
사람들이 image를 uploading하였다. (cat) 오늘의 dropout mask는 cat을 잘 분류하는데
내일의 dropout mask는 cat을 없애버릴 수도 있다. 
이것이 바로 test time에서의 dropout randomness가 유발하게 되는 문제점이다.
따라서 test time엔 average out 을 사용한다.
test시 그냥 dropout mask probability를 곱해줘버린다. 
(train시는 drop시켜버리는데)

##### data argumentation

가지고 있는 data를 조금 변형시켜서 추가로 넣어서 사용한다.
(data의 뻥튀기…?)
crop / scale 등의 방법을 사용 (Linear한 방법내에서 적용?)
contrast / brightness도 건드리는 경우가 있다.
translation, rotation, shearing, .... 방법이 많다.

##### drop connect

##### Fractional Max Pooling

먼저 Pooling은 data내 Pixel을 기준, 일정부분을 crop하여 (crop이라고 하니까 좀 이상하긴 한데)
그 부분내 가장 큰값을 해당 Pixel에 부여하는 방법이다.
해당 idea는 좀 봐야할듯.

##### Stochastic Depth

최근에 나온 방법. very deep network에서 사용.
drop out과 비슷한 방법인듯...?

-----



#### Transfer learning

small data set을 학습하면 data가 좀 부족.
최초 parameter값을 이미 훈련한 dataset(large data)으로부터 가져오는 건듯.
여튼 기존 dataset의 parameter를 가져오는 것이
내가 사용할 data에 따라 다른 transfer learning 전략이 필요하다.



#### Reference

* http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
* http://enginius.tistory.com/476
* http://incredible.ai/artificial-intelligence/2017/05/13/Transfer-Learning/ (trasnfer learning)

