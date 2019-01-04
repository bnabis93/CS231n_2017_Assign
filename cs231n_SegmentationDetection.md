## CS231n Detection and Segmentation

### Index

* #### Segmentation

* #### Localization

* #### Detection

----

### Sementic segmentation
Input (image) => output (이미지로 부터 얻어진 결과인 decision , category)
그런데 이러한 output을 전체 image로부터 결정하는 것이 아니라, 모든 픽셀에 대하여 decision을 얻는다.

#####<img src="/Users/hyeonwoojeong/Documents/cat_sementicSeg.png"  style="zoom:30%">
위의 고양이 사진을 classify 하려고 한다. 육안으로 Cat, Grass, Flower,background class를 분류 할 수 있다. 이는 각 Pixel 단위의 분석으로 이루어 지게 된다. (이전 bounding box들과는 다름)
즉, 이미지를 pixel단위로 분류를 해 내는 기술이 sementic segmentation이다.

#### Semantic Segmentation Idea
* ##### Sliding Window

  작은 단위의 window를 생성하고 해당 단위로 전체 이미지를 crop한다. (extract patch)
  기존에 전체 이미지를 보아 class 를 나누는 작업에서, 전체 이미지의 한 부분을 class로 나누는 작업으로 변화하게 됨. (하지만 연산량이 굉장히 많아지게 됨, 각 crop image마다 class를 나누어주기 때문)

* ##### Fully Convolutional metod

  기존 좋은 성능을 가지는 네트워크 (VGG, GoogleNet, etc...) 등을 이용하는 방법이다. 
  해당 방법의 뒷부분(network의 뒷부분)에는 fully conneted layer가 존재한다. 하지만 해당 layer는 **<u>고정된 크기의 입력</u>**만을 받아들이고 **<u>위치정보가 사라진다</u>**는 단점이 존재한다. 이를 어떻게 Sementic segmentation에 이용하는 것 일까?
  [바로 fully connected layer를 1*1 convolution으로 보면 해결이 된다.][1]  -[1]
  이렇게 하게되면 앞서 말한 위치정보가 사라지는 단점을 해결 할 수 있게 된다.
  (전체 이미지를 처리하는 것이기 때문에 속도도 향상되는데, 이해가 잘 안간다.)

  #####<img src="/Users/hyeonwoojeong/Desktop/sementicSegmentation_ex02.jpg" style="zoom:30%">
  <위의 이미지는 FCN을 이용한 Sementic segmentation 예이다.>

  * **Downsamping**
    Stride / pooing에 의한 spatial 한 size의 감소 
  * **Upsampling**

    여러단계의 convoultion을 거치게 되면 feature-map의 크기가 줄어들게 된다.(**Down Sampling** 된다, 위의 예에서도 점점 줄어드는 것을 관찰 할 수있다.) 따라서 위의 예의 3번과 같이 줄어든 size를 다시 키우는 과정이 필요하다.
    단순한 interpolation의 방법으로 써 image의 크기를 키워도 되지만, 이 역시도 고정된 값이 아닌, 학습을 통하여 결정을 하는 것이 좋다.

    * **unpooling**

      #####<img src="/Users/hyeonwoojeong/Desktop/스크린샷 2018-11-23 오전 11.16.50.png" style="zoom:80%">
      해당 방법은 이미지를 보는 것이 설명이 더 빠를 것 같다.
      일종의 interpolation 방법이다. 같은 수로 채우거나 0으로 채우거나.
      (매우 원시적인 방법이라 생각한다.)

    * **Max Unpooling**
      #####![image-20181123112406540](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181123112406540.png)
      이것또한 이미지로 이해를 하는것이 빠르다.
      Max Pooling시의 position을 기억해놓았다가, Max Unpooling시 해당 position에 복구를 해주는 방법이다.
      이것이 왜 좋은 Idea인가?
      => Sementic segmentation의 목적은 기존 edge등의 feature를 보는 것이 아닌 Pixel을 보아 분류를 하게 된다. (detail한 분류를 하고자 한다.) max pooling시 사라진 공간정보를 Max unpooling을 하게 됨으로써 복구를 한다는 의의가 존재한다.

    * **Learnable upsampling : Transpose Convolution** (Deconvolution)
      앞선 방법은 parameter를 update하는 (learning) 방식이 아니었다.
      우리의 목적은 최대한 잘 보존된 공간 정보로써 이전의 정보를 복구하는 것 이다.
      먼저 convolution 연산을 잘 살펴보자.
      convolution은 input과 kernel, 그리고 이들의 element-wise 곱의 합으로 만들어지는 output으로 이루어 진다.
      
      #####<img src="/Users/hyeonwoojeong/Desktop/sementicSegmentation_03.JPG" style="zoom:40%">
      
      [이러한 convolution 연산은 input matrix와 output matrix 사이에 위치연결성(positional connectivity)가 존재한다는 것 이다.][2] -[2]
      잘 이해가 안가는데, 3x3 kernel의 경우 9개의 input 이 하나의 output을 이루는 many-to-one 관계를 형성하게 되는데, 이러한 관계를 위치 연결성이라고 한다.
      이를 반대로 하는 것이 transpose convolution(deconvolution)이다. 즉, one-to-many 관계를 이루게 된다. 3x3 kernel의 경우 하나가 9개와 관계를 이루고 있는다.

      3x3 kernel -> 4x16 kernel로 재배치하겠다. (갑작스럽지만 일단 읽자)

      #####<img src="/Users/hyeonwoojeong/Desktop/sementic04.png" style="zoom:30%">

      #####<img src="/Users/hyeonwoojeong/Desktop/sementic05.png" style="zoom:40%">
      
      그림을 보면 3x3 kernel을 4x16 kernel로 재배치를 한 것을 볼 수 있고, 4x16 kernel의 row에 기존 3x3 kernel 정보가 들어있는것을 확인 할 수 있다.
      이러한 개념을 이용하여 input matrix를 재배치 할 수 있는데, 다음과 같이 재배치를 할 것 이다.

      #####<img src="/Users/hyeonwoojeong/Desktop/sementic06.png" style="zoom:30%">
      4x4 input matrix => 1x16 matrix로 재배치를 할 수 있고 이를 위에서 재배치한 convolution layer와 행렬곱 연산을 할 수 있다. (이때 Transpose를 해주게 된다.)
      (4x16) x (1x16).T = 4x1 output matrix 
      4x1 output matrix 를 2x2 output matrix로 재배치 할 수 있다.
      즉, 행렬간의 재배치를 통하여 원하는 형태의 행렬 matrix를 만들 수 있다는 것이 요지이다.

      우리는 2x2 matrix(input)에서 4x4 matrix(output)로 upsamping을 하는 것이 목표이다. 
      따라서 16x4 matrix와 convolution을 하게되면 (2x2 => 4x1상태) 원하는 4x4 행렬을 만들 수 있게 된다.

      ![sementic08](/Users/hyeonwoojeong/Desktop/sementic08.png "style"=zoom:30%"")
      
      위의 그림을 보면 16x1행렬을 만들었고, 이를 재배치하여 4x4 행렬을 만들면 된다.
      upsamping이 되었다.

      좀 더 직관적인 그림을 보자

      #####![sementic09](/Users/hyeonwoojeong/Desktop/sementic09.jpg)
      
      #####![sementic10](/Users/hyeonwoojeong/Desktop/sementic10.jpg)
      <[그림][3]은  [3]에서 가져옴>
      upsampling을 시키는 것이고, weight도 update가 된다.
      transpose convolution 연산은 convolution의 backpropagation 연산과 동일하다. [3]
      어렵다… 직관적으로는 알….거같긴하다...

-----
### Classification + localization
#####<img src="/Users/hyeonwoojeong/Documents/cat_sementicSeg.png"  style="zoom:30%">
'Cat' class를 찾고, 그 주위에 bounding box를 그리고자 한다. 어떻게 해야 할까?

##### Localization (용어 설명)

[bounding box를 통하여 물체의 존재 영역을 파악하는 행위를 말한다.][4] -[4]
최대 5개의 후보 결과를 도출 해 내고, Ground-truth와 비교하여 50% 이상 영역이 일치하면 맞다고 본다.

즉, Classification을 이용하여 해당 object를 분류하고, localization하여 종국에는 object를 detection하게 된다.

#####![image-20181123142347976](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181123142347976.png "style="zoom:10%"")

class 분류 + localization 을 같이하는 model, 2 loss가 존재한다. 
강의에서 예를 들어준 Human pose estimation이 좋은 예 인듯.
class 분류 + joint (localization)

-----

### Object Detection

#####![objectDetection01](/Users/hyeonwoojeong/Documents/objectDetection01.png)

#####<img src="/Users/hyeonwoojeong/Documents/cat_sementicSeg.png"  style="zoom:30%">

object detection 에서는 같은 model 을 써도 output의 갯수가 달라진다.

#### Method

* ##### Sliding Window

  위에서 설명한 방법과 같다.
  image를 쪼개어 학습시키고 검출한다. (결국엔 전체 이미지를 보게 된다.)
  검출하고자 하는 객체와 Background라는 class로 나뉘게 된다.
  (역시나 연산량이 굉장히 많아지게 된다.)

* ##### Region Proposal (고전적인 computer vision 방법에 가깝다)

  object가 있을 만한 곳만 본다.
  #####![img](https://i1.wp.com/junn.in/wp-content/uploads/2018/04/6.png?resize=1170%2C488&ssl=1) 
  [Selective search를 통하여 텍스쳐나 색, 강도 등이 유사한 픽세리리 연결된 Window를 만들고 이를 본다.][5] -[5], R-CNN의 아이디어 이다.
   #####![image-20181123145213717](/Users/hyeonwoojeong/Library/Application Support/typora-user-images/image-20181123145213717.png)
   <R-CNN의 구조 및 방법>
  그래도 여전히 연산량은 굉장히 많다. (약 2000개의 region proposal 이 존재하므로)

* **Fast R-CNN**
  CNN을 한 이미지에 딱 한번만 돌리고, 2000개의 region proposal 을 나눠서 모델에 넣지 않고 계산된 값을 공유하는 방법 (ROI pooling)

* #####Faster R-CNN
  #####<img src="/Users/hyeonwoojeong/Desktop/objectDetection02.png" style="zoom:50%">
  [CNN 결과를 selective search 대신에 region proposal에 이용하자.][5] -[5]
  그리고 model을 다 합쳐버렸다. (multi-tasking learning)

* ##### Yolo(you look only once)

  이미지 내의 bounding box와 class probability를 single regression problem으로 간주하여 이미지를 한번 보는 것으로 object의 종류와 위치를 얻어낸다.
  [yolo][6] - [6] 을 읽어보는 것이 이해가 더 빠를듯 하다.
  간단히 말하면 (참고로 하나의 network만 사용)

  1. input을 SxS grid로 나눈다.
  2. 각 grid cell을 B개의 bounding box와 각 bounding box에 대한 confidence score를 가진다.
  3. 각 grid cell은 C개의 conditional class probability를 가ㅣㄴ다.
  4. 각 bounding box는 (x,y,h,w, confidence)로 구성된다.

  #####![img](https://curt-park.github.io/images/yolo/Figure2.JPG)

* #####SSD (single shot detection)

  [yolo와 같이 하나의 network를 사용하여 bounding box와 class를 찾는다.][7] -[7]

  yolo와의 차이점은 yolo는 최종 feature map에만 bounding box와 class정보가 있지만
  SSD는 여러 hidden layer에 정보가 분산되어 있다.

----

#### Object Detection + Captioning

##### Dense Captioning

keyword만 알아두자.

----

#### Instance Segmentation

##### Sementic segmentation + object detection 이라 볼 수 있다.

pixel단위의 정확성을 가지고, 해당 객체가 무엇인지 (multi class?̊̈) 판단하는 문제이다.
![img](https://i0.wp.com/hugrypiggykim.com/wp-content/uploads/2018/03/4_fcn.png?resize=636%2C345)
<해당 그림을 보면 바로 이해가 됨> [[8][8]]

[Mask R-CNN][8] -[8] 
Mask R-CNN으로 joint position estimation도 가능.



-----

#### Reference
* [1] : https://laonple.blog.me/220752877630
* [2] : https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/
* [3] : https://www.facebook.com/groups/TensorFlowKR/permalink/576593562681706/?comment_id=576605562680506&reply_comment_id=576606212680441
* [4] : http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220752877630&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
* [5] : https://junn.in/archives/2517
* [6] : https://curt-park.github.io/2017-03-26/yolo/
* [7] : https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=221007697796&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F
* [8] : http://hugrypiggykim.com/2018/03/26/mask-r-cnn/

[1]: https://laonple.blog.me/220752877630
[2]: https://zzsza.github.io/data/2018/06/25/upsampling-with-transposed-convolution/
[3]: https://www.facebook.com/groups/TensorFlowKR/permalink/576593562681706/?comment_id=576605562680506&reply_comment_id=576606212680441
[4]: http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220752877630&categoryNo=0&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
[5]: https://junn.in/archives/2517
[6]: https://curt-park.github.io/2017-03-26/yolo/
[7]: https://m.blog.naver.com/PostView.nhn?blogId=sogangori&logNo=221007697796&proxyReferer=https%3A%2F%2Fwww.google.co.kr%2F
[8]: http://hugrypiggykim.com/2018/03/26/mask-r-cnn/


