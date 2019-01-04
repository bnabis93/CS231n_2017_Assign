## Cs231n, Visualizing and Understanding

#### Index

* **How to train ConvNet?̊̈ (inside ConvNet)**
  ConvNet의 내부는 어떻게 돌아가는 것인가?



----

#### CNN Review
#####![cs231n_VisualAndUnderstaning](/Users/hyeonwoojeong/Desktop/cs231n_VisualAndUnderstaning.gif)

위는 Input에 대하여 Convolution이 어떻게 진행이 되는지를 보여주는 그림이다.
$\begin{bmatrix} 1 & 0 &1 \\ 0 & 1 & 0 \\ 1 & 0 & 1\end{bmatrix} $ 의 convolution matrix가 존재하고 
Image를 convolution matrix와 convolution하여 **Feature Map**(Convolved Feature)을 만들게 된다.
결과를 보면, 기존 이미지에 비해 크기가 줄어듦을 알 수 있다. 이는 중요한 포인트니 기억해두도록 하자.
(이를 막기위하여 padding이 존재)

#### Pooling Layer

pooling layer는 convolution layer의 output**(Activation Map)**을 input으로 받는다.
convolution의 결과로 출력된 데이터(activation map)의 크기를 줄이거나, 강조를 하기위하여(Max pooling 등) 사용된다.
Pooling layer는 다음과 같은 특징을 가진다.

* 학습할 Parameter가 따로 없다.
* Pooling시 행렬(image등) 의 크기가 줄어든다
* 채널수는 유지된다.
![Pooling ìì : Max Pooling, Average Pooling](https://taewanmerepo.github.io/2018/02/cnn/maxpulling.png)

#### Layer 별 출력되는 데이터의 크기

##### Convolution layer

Filter의 크기, Stride 정도에 따라 Feature Map의 크기가 결정된다.

* 입력데이터 : H(높이) x W(폭)
* 필터 : FH(높이) x FW(폭)
* Strid 크기 : S
* Padding : P

$ Output Height = OH = {(H+2P - FH)\over S} +1$

$OutputWeight = OW = {(W+2P-FW)\over S} +1$

식의 결과는 자연수이고, Convolution Layer -> Pooling Layer가 오게되는 경우에,
Feature Map의 크기는 Pooling 크기의 배수여야 한다.
Pooling 사이즈가 3x3이면 (pooling output이 3x3) Feature Map(Convolution layer 출력) 은 3의 배수여야 한다.
따라서 설계를 할 때 해당 조건을 만족하도록 설계해야 한다.

##### Pooling layer
$OutputPoolingRow = {InputRowSize\over Pooling Size }$ 

$OutputColumnSize = {InputColumnSize \over PoolingSize}$

따라서 자연수가 나오기 위해서는 위에서 말한 조건인, FeatureMap이 Pooling size의 배수가 되어야 한다.



#### CNN Parameter

![ìì  CNN ì´ë¯¸ì§](https://taewanmerepo.github.io/2018/01/cnn/cnn.png)

4개의 Convolution Layer, 3개의 Pooling layer가 존재한다.

* 최초 input 39x31x1
* 4x4 convolution kernel , stride = 1
* 2x2 pooling kernel, stride = 2

##### 이때의 Parameter는 어떻게 계산할까?

Convolution layer, Pooling layer에서의 Parameter 계산을 해보겠다. (하나씩만)



##### Convolution Layer Parameter

* 입력 = 39x31x1
* filter  = 4x4x20 , 4x4 filter 20개
* stride = 1

이로부터 얻어지는 Activation Map (Convolution Layer Output) 의 크기는 다음과 같다.
$Activation\  Map \ Row = {Input\ Row - Filter \ Row\over stride}+1  = {39-4\over1}+1 = 36$

$Activation\  Map \ Col = {Input\ Col - Filter \ Col\over stride}+1  = {31-4\over1}+1 = 28$

Activation Map Size = (36,28,20) 이다. (Filter 의 갯수가 20개)
학습 대상이 되는 Parameter는 Filter 이다. (입력이 학습이 될 수는 없으니)
따라서 **학습 파라미터의 갯수는 4x4x20  = 320** 이다.

* Activation Map(output) = (36,28,20)
* 학습 Parameter = 4x4x20 = 320



##### Pooling Layer Parameter

Pooling시 별도의 학습 Parameter가 필요하지 않다. 
(단순한 Convolution 연산이기 때문에)
하지만 출력 크기는 달라지게 된다. (Pooling시 작아지게 된다.)

$Pooling\ Out\ Row = {Input\ Row \over Pooling \ Size}= {36\over2} = 18 $

$Pooling\ Out\ Col = {Input\ Col \over Pooling \ Size} = {28\over2} = 14$

* 출력 데이터 = (18,14,20)
* 학습 Parameter 없음

이런식으로 Parameter를 계산하면 된다.

Fully Connected Layer의 경우 입력노드 x 출력노드 수만큼의 Parameter가 존재한다.
(단순한 Neural Net 구조만 생각해도 답이 나온다.)



-----

### cs231n

#### Visualizing

##### First Layer

input에 대한 최초 Layer를 Visualizing 한 결과

#####![image-20181129110450997](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181129110450997.png)

우리는 Oriented edge(방향성이 있는 edge)를 관찰 할 수 있다. (Feature Map)
(다양한 edge와 색을 가지는 edge들이 보인다.)



#####Last Layer

Alex Net 의 경우 Last layer가 FC(Fully connect) 형태이다.
이는 이전에 만든 feature map을 합친다고 할 수 있다.

##### Nearest Neighbor

우리는 굉장히 흥미로운 결과를 관찰 할 수 있다.
![image-20181129205659932](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181129205659932.png)
분명 두 고양이는 pixel 수준의 비교에서는 다르다 (한 고양이는 뛰고 있고, 다른 고양이는 앉아 있다.)
여기서 pixel 수준의 비교는 단순 두 pixel의 차를 의미한다.
하지만 network를 돌려 학습을 시킨다면, 이 둘은 같은 class로 분류 될 것 이다.
 (이 둘의 semantic content는 같다.)

##### Dimensionality Reduction, 차원 축소

=> PCA를 이용하여 차원을 줄일 수 있음. (혹은 t-SNE)
Neural net으로 얻어진 수많은 feature 중 중요한 것만 남기게 된다.
PCA와 t-SNE 관련해서 한번 정리해야할듯. 

##### Maximally Activating Patches

Convolutional layer를 거치면 많은 feature map이 나오는데, 이중 가장 큰값을 가져온다는 말인듯.
(맞나...?)

##### Occlusion Experiments

이미지의 한 부분을 masking (가리고) 하고 학습을 시킨다. (이때 가린 부분은 이미지의 평균값으로 채운다.)
이들의 가정은 input image 의 일부분을 가리면 결과값이 상당히 바뀔것이라 예상하였다.
이 실험의 결과로 어떠한 픽셀이 classification decision에 영향을 주는지 유추 할 수 있게 됨…

##### Saliency Maps

이또한 어느 pixel이 classification에 영향을 주기 알아보기 위한 실험
이미지의 각 픽셀의 특성을 반영한 또 다른 이미지를 생성함.
즉, 원본 이미지를 변형하거나 단순화하여 특정 부분만을 본다든지 혹은 어떠한 부분에 대한 분석 등을 가능하게 함. (밑의 그림을 보면 그림을 굉장히 단순화 시킴.)
![image-20181129213439028](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181129213439028.png)
이를 Sementic segmentation에 이용 할 수도 있다. (labeled data가 필요없이!)
Grabcut Segmentation Algorithm을 함께 사용한다. (이런게 있다는것만 알고 있자.)

##### Guided backprop

positive gradient만을 visualizing 한것.
propagating시 negative한 gradient는 모조리 0으로 만들어 버린다.
그러면 영향을 미치는 (특징에 대하여) gradient를 더 잘 시각화 할 수 있겠지.



##### Gradient Ascent

$I = max(f(I) + R(I))$ ,f(I) = Neuron value, R(I) = Natural image regularizer
Regularizer는 overfitting을 방지하기 위함이다.

기존에 하던 gradient decent 와는 달리 gradient를 최대한 크게 만들어줘본다
정확히는 극소점을 찾는 decent와는 달리, ascent는 극댓값을 찾는다고 생각하면 된다.

Neuron value의 최댓값을 찾으면서도, Natural image(원본 이미지)와 유사한 형태의 이미지를 얻길 원한다.

듣다보니 GAN의 초기형태를 보는 것만 같다.

* L2 regularizer를 이용한 gradient ascent

  #####![image-20181130141117833](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181130141117833.png)
  그림을 자세히 보면 윤곽정도는 알아 볼 수 있다. (정말로?̊̈) 희미하게 비슷한 pattern이 보이는것 같다.

* Better regularizer
  L2 norem 에다가 추가적으로 더 Regularize를 하였다.

  1. Gasussian Blur
  2. Clip pixel with small value -> 0
  3. Clip pixel with small gradient -> 0
  #####![image-20181130141621072](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181130141621072.png)
  더 나은 결과를 보여주고 있다.

##### Fooling images

1. randomly한 image로 부터 시작하고, class를 randomly 하게 뽑는다.
2. class 를 maximize 하도록 이미지를 수정한다..?
3. Loop (network is fooled)

Ian goodfellow가 더 자세히 설명해줄거라고 한다… 
정확히 이해가 가지는 않는다.

#### Deep dream

기존 특정 neuron 을 maximize 시키는 방법과 달리, 특정 계층에서의 neuron activation을 증폭시켰다.
Choose image and a layer in a CNN; repeat:

1. [Forward] compute activation (선택된 layer에 대해)
2. 선택된 layer의 gradient를 activation 과 같게 만들어준다 (?)
3. [Backward] comute gradient on image
4. Update image

gradient를 계산하기전에 jitter image(noise image)를 만들어준다.
음… 코드와 논문을 보고 이해하는게 빠를 것 같기도 하

##### Feature Inversion

주어진 CNN feature vector를 바탕으로 새로운 image를 찾는다. (reconstructing image.)
이를 이해하기 전에 Texture Synthesis를 이해하고 가자.



#### Texture Synsthesis
다음과 같은 texture를 가진 input(small patch)이 주어졌을 때, 어떻게 하면 같은 texture를 가지면서 더 큰 이미지로 만들어 낼수 있을까?
#####![image-20181130144718316](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181130144718316.png)

* **고전적인 접근법(Computer graphics), Nearest Neighbor**
  input 이미지가 있으면 해당 이미지를 계속해서 copy하여  (그래서 Nearest neighbor라는 말인듯!)
  texture를 유지하며 거대한 output을 만들어낸다.
  하지만 textrue가 복잡해지면, 위와 같은 단순한 방법으로는 texture를 유지하며 크기를 키울 수 없다.
  다른 방법이 필요하다.
* **Neural Texuture Synthesis : Gram Matrix**
  * Gram matrix
    그림의 texture를 알기 위해서는 전체의 feature 정보를 알고, 이를 비교해야한다. 
    이는 flatten feature map간의 covariance라고 하는데… 아직 잘 모르겠다.
    그런데, covariance matrix와의 차이점이 무엇인가?
    일단, covariance matrix의 계산량이 많아 사용하지 않는다고 강의에서는 말한다. 알아보자.
    홍정모 교수님의 강의를 참고하도록 하자. (Gram matrix)
    http://blog.naver.com/PostView.nhn?blogId=atelierjpro&logNo=221180412283&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
  * Neural Texture synthesis
    VGG-19 model 사용, Gram matrix 계산
    random noise로 부터 image 생성
    그 후, loos 계산, Back prop, 반복

결과로 Deep dream, style transfer 등이 등장함

#### Style transfer

크게 세부분으로 나뉜다.

* Content Image $y_{c}$
* Style image $y_{s}$ , 따라하고 싶은 이미지
* output image (start with noise)

이를 바탕으로 학습을 하여 우리가 아는 그 유명한 고흐 스타일의 그림이 나오게 된다.
multiple gram matrix로 여러 style으 따라하기도 한다. (Deep dream)


----

receptive field : **필터가 한번 보는 영역**으로 사진의 feature를 추출하기 위해선 receptive field가 높을수록 좋다.

----

#### Reference

* [1] : http://taewan.kim/post/cnn/ , CNN의 이해
* https://zzsza.github.io/data/2018/02/23/introduction-convolution/ receptive field
* http://bcho.tistory.com/1210 , t-sne
* http://blog.naver.com/PostView.nhn?blogId=atelierjpro&logNo=221180412283&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView , about gram matrix