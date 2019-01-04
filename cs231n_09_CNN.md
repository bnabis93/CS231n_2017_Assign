### CS231n, CNN Architectures

----

* #### Model(case study)

  * VGG
  * AlexNet
  * etc...

----

#### AlexNet (2012)

first large scale CNN architecture.
input =96*  (227 * 227 * 3)
first layer = 11* 11, stride = 4
=> first output layer = (227 - 11) / 4 +1
total number of parameter in this layer?
=> (11 * 11 * 3) * 96

Pooling layer에서는 parameter가 존재하지 않는다.
여기서 말하는 parameter는 weight이다.
Pooling은 단순히 kernal내에서 max값을 뽑아내는 것이기 떄문에 parameter가 존재하지 않는다.

AlexNet을 보면 끝에서 FCN으로 변환하는게 인상깊었다.
ReLU사용했다.
그리고 Data argumentation을 굉장히 중요시 여기는 듯. (일종의 전처리)

Layer를 나눈건 당시의 하드웨어 한계(GPU) 때문이다.

#### VGG Net (2014)

AlexNet에 비해 깊어진 구조를 가지게 되었다.
하지만 그보다 작은 filter를 사용하였고, (3 * 3) 필터의 사이즈도 고정된 형태이다.
필터에 대한 직관이 필요하다. (필터의 변화에 따라 무엇이 변화하는가)
3개의 연속적인 3 * 3 filter는 하나의 7 * 7 filter와 같은 효과를 낸다.
=> 더 적은 parameter로 같은 효과를 낼 수 있다.
filter의 깊이가 깊어진다는것은...?̊̈ 

#### GoogleNet (2014)

좀더 깊어진 구조.
**inception module**… **(뭐지?)**
network내부에 존재하는 또다른 network.
inception module 하나가 전체 구조에서 보면 하나의 Layer인데,
가까이서 보게되면 inception module이 그 안에서 내부적으로 하나의 소형 network를 이루고 있음

**Naive inception Module**
내부적으로 filter / pooling layer가 존재하는 구조. (병렬처리)
계산량이 넘나 많아짐.
그렇다면 해당 module의 output은 어떻게 나와야 하는가?
=> 모든 depth를 합친 (filter concatenation, filter 연결.)

##### inception module with dimension reduction

1 * 1 filter를 사용하여 depth를 줄여 연산량을 줄였다.

No FC Layes **(why?)**
parameter가 상당히 줄어들었다.

##### Auxiliary stem layer

googlenet을 보면 처음과 중간에 기존에 사용하던 layer가 존재함을 볼 수 있다.
이는 왜 존재하는 것인가.
중간중간 결과를 낸다는 의미인것같긴한데.

#### Resnet (2015)

갑자기 엄청 깊어짐… (152 Layer, google net은 20 layer정도)
단순히 layer를 깊게 쌓는다고 해서 결과가 좋아지는 것은 아니다.
(56 layer - 20 layer의 비교를 통한 결과, because of overfitting)
=> 이러한 결과가 생기는 것은 optimization 문제이다!  이를 해결하면 Deep 한 Network에서 효과가 좋을것
(Resnet의 가설)
=> resiual mapping을 통한 해결 (not directly)
**residual mapping**
activation (Relu등)을 통하여 얻어진 값을 학습하는 것이 아니라,
input과 output의 차이인 residual 을 학습는 방식.

#### Network in Network(2014)

#### Identity Mapping in Deep Residual NN (2016)

residual block 이용

#### Wide Residual Networks

filter를 k개 사용. (deep filter...?)

#### Aggregated Residual ~ (2016)

residual block이 병렬적으로 여러개 존재.

#### Stochastic Depth(2016)

많이 들어본 아이디어. 
residual block을 중간중간 random하게 끄는건가

#### Not using residual.

* #### FractalNet(2017)

* #### DenseNet(2017)

  good feature map...

* #### SqueezeNet(2017)

  정확도는 낮지만 parameter수가 굉장히 줄었다.

##### Stride

stride는 다음 pixel 로 갈때의 간격을 의미한다.



#### Reference

* http://tmmse.xyz/2016/10/15/lstm-resnet/
* https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=220793640991&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F