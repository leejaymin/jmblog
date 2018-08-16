---
layout: post
title:  "Loss functions"
date:   2018-08-15 11:28:24 +0900
categories: jekyll update
---

# Loss functions #

----------

딥러닝 하다보면 그냥 사용했던 `Loss Function`들에 대해서 한번 다뤄 본다.

## Entropy ##

정보를 최적으로 인코딩하기 위해 필요한 `bit`의 수
$\log_{2}^{}{N}$

### 요일 예시 (Uniform) ###

만약 요일에 대한 정보를 메시지에 실어서 보내야 한다면 가장 최소의 bit를 사용한다면 얼마나 될까?
`Yes`/`No`수준이라면 그냥 1 bit만으로 충분하다.

그렇다면 econding을 day of the week을 해야한다고하면, 7 values를 encode 해야한다.
결국 $\log_{2}^{}{7}$
`월화수..일` 7가지를 bit로 전송하기 위해선 `3bit`필요 하게 된다.
- bits (000 – Monday, 001 – Tuesday, …, 110 – Sunday, 111- unused).
> bit의 발생 빈도가 uniform 하다는 가정에 의해서 발생

### Speech of a sequence of words (skewness) ###

만약 영어 단어를 말하는것을 encode 하는 것을 생각해 보자.
그냥 단순히 uniform하다고 가정하면 $\log_{2}^{}{40} = 5.3$ 이다.

하지만 4개의 `Tag`값들이 `90%`가 발생한다면 좀 더 효율적인 `bit encode`를 만들어 낼 수 있다.

40개의 문자가 있고 상위 4개가 90%로 발생한다.
- `ART`, `P`, `N`, `V`

아이디어는 첫 번째 `bit`를 단순히 위 네개의 TAG를 구분하는데 사용 한다.
- YES 이면 추가로 `2bit`를 더 필요로 해서 구분하게 된다.
- NO 이면 `6bit`를 더 필요로 한다. `40-4 = 36`이고, $\log_{2}^{}{36}=5.16$ 

정리하면 아래 표와 같다.
![](https://i.imgur.com/Kuvrqxr.jpg)

이것을 생각한 방식으로 계산하면 아래와 같다.
필요한 bit수 계산 = $0.9 \times 3 + 0.1 \times 7 = 3.4$

원래 아이디어 보다 `bit`수가 더 줄어 들었다. 거의 반만큼.
여기서 더 최적화 할 수 있을까?

Information theory provides an answer.
> As mentioned before, `Entropy` is a measure of randomness in a probability distribution.
> A central theorem of information theory states that the entropy of `p` specifies the minimum number of bits needed to encode the values of a random variable `X` with probability function `p`.


### Definition of Entropy ###

`X`를 `random variable`이라고 하면 `p`는 probability function이다. 
$$ p(x_{i}) = P(X=x_{i}) $$
이라고 한다 (보통 algebra variable과 혼동하지 않기 위해서 capital로 표기하지만 여기선 간소화를 위해서 assumption을 만들고 시작).

Entropy of X (or p)의 정의는 아래와같다.

$H(X) = H(p) = - \sum_{i}{p(x_i) \times \log{p(x_i)} }$, where $x_{i}$ ranges over the vocabulary of $X$

위 식으로 다시 계산해보면 TAG `random variable`에 대해서 actual probability distribution을 표1에서와 같이 알기 때문에 TAG의 entropy는 아래와 같이 계산된다.

$ H(TAG) = -(4 \times (.225 \times \log_{2}{.225}) + 36 \times (.0028 \times \log_{2}{.0028})) = -(-1.04 + -.82 + -.85) = 2.72 $

결과 값을 통해서 그냥 단순하게 coding 했던것 보다 좋았던 `3.4`보다 더 낮은 `2.72`가 optimal 이라는것을 알게 되었다.
하지만, `entropy`의 문제는 이게 가장 best한 conding 디자인이 아니라는것만 알려주지 실제로 그래서 어떻게 coding 해야 optimal에 도달하는지를 알려주지 않는다. 더 나아가 physically이게 가능한 것인지도 알 수 없다. 


**Data science에서 보는 값에 따른 해석**
- `High Entropy`는 x가 uniform한 distribution을 가지게 된다. 즉 boring하다고 해석할수도 있고 diversity가 높다고 할수도 있다. Active Learning에서 그래서 이러한 Entropy를 이용해서 smapling을 할지 말지를 고려할 수도 있다.
- `Low Entropy`는 skewness가 높다고 할 수 있다. 결국 varied distribution이라 할 수 있고 `peak`나 `valley`가 많다고 할 수 있다.


## Cross-Entropy ##

두개의 probability function을 비교하는것이 cross entropy이다.
- `p`와 `q`로 구분한다.
이러한 `Cross Entropy`는 symmetric하지 않기 때문에 그 역은 성립하지 않는다.
`p`를 보통 target으로 `q`를 estimated one으로 설정한다.
그렇게 비교해서 얼마나 서로 가까운지를 알아내서 그것을 최적화하는 방법으로 학습에 사용한다.

$$H(p,q)= -\sum _{ i }{ p(x_{ i }) } \log { q(x_{ i }) } $$

$p_{y=1} = y$ 이고 $p_{y=0} = 1-y$ 라면,  $p \in (y, 1-y), q \in (\hat{y} ,1-\hat{y})$

$$H(p,q)= -\sum _{ i }{ p(x_{ i }) } \log { q(x_{ i }) } = -y\ln{\hat{y}} + (1-y)\ln(1-\hat{y})$$


**MSE의 문제점**
$$MSE = C = \frac{{(y-a)^2}}{2}$$
$$a = \sigma(z)$$
$$z = wx+b$$
위의 것을 미분하면,
$$ \frac{\partial C}{\partial w} = (a-y)\sigma^{\prime}  {(z)} x $$
$\sigma^{\prime}  {(z)}$ 값이 최대 `0.25`가 나오고 대부분 0이기 때문에 vanishing gradient가 생기는건 잘알려진 사실이다.

**cross entropy 미분**

$$ C= -\frac{1}{n} \sum_{x}^{}{y\ln{a} + (1-y)\ln(1-a)} $$

$$\frac { \partial C }{ \partial w_{ j } } =-\frac { 1 }{ n } \sum _{ x }{ (\frac { y }{ \sigma (z) } -\frac { (1-y) }{ 1-\sigma (z) } ) } \frac { \partial \sigma  }{ \partial w_{ j } } \\ =-\frac { 1 }{ n } \sum _{ x }{ (\frac { y }{ \sigma (z) } -\frac { (1-y) }{ 1-\sigma (z) } ) } \sigma ^{ \prime  }(z)x_{ j }\\ =\frac { 1 }{ n } \sum _{ x }{ \frac { \sigma ^{ \prime  }(z)x_{ j } }{ \sigma (z)(1-\sigma (z)) }  } \sigma ^{ \prime  }(z)-y $$


최종적으로 아래식 처럼 $\sigma(z)$가 살아있기 때문에 vanishing 문제를 적게 발생 시키는 좋은 폼을 형성한다.
$$\frac{\partial C}{\partial w_{j}} = \frac{1}{n} \sum_{x}^{}{x_{j}(\sigma(z)-y) }$$


## KL-divergence ##

두 분포 차이를 줄이기 위해서는 `KL-Divergence`를 사용한다.

$$D_{KL}(P \parallel Q) = -\sum_{x}^{}{p(x) \log{q(x)}} + \sum_{x}^{}{p(x) \log{p(x)}}$$

$$ = H(P,Q) - H(P)$$


## 참고자료 ##
[Lecture6: Using Entropy for Evaluating and Comparing Probability Distributions](http://www.cs.rochester.edu/u/james/CSC248/Lec6.pdf)

[정보량을 나타내는 앤트로피(Entropy)](https://www.youtube.com/watch?v=zJmbkp9TCXY&list=PL0oFI08O71gKEXITQ7OG2SCCXkrtid7Fq&index=21)

[4.1. Cross entropy와 KL divergence](https://www.youtube.com/watch?v=uMYhthKw1PU)
