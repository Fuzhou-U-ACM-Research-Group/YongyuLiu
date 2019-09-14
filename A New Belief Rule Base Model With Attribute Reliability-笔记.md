## 1、研究目的
&emsp;&emsp;现有的BRB模型都是直接使用输入数据作为训练数据，但在现实中获取得到的数据往往不能保证是完全正确的，特别是在工程领域中的数据，工程实践中的数据会受到两个因素影响：长时间工作的传感器和实际工作环境的噪声。这两个因素会使得观测到的数据不准确，进而影响通过数据建立BRB模型的准确性。所以，这篇论文针对上述问题研究输入参数的可靠性，提出BRB-r模型来解决上述问题。
## 2、使用方法
&emsp;&emsp;BRB-r模型的建立主要分为四个步骤：
- step1:从传感器中获取到观测数据$(x_1^*,\ldots,x_i^*,\ldots,x_M^*)$;
- step2:通过公差范围（tolerance range）计算参数可靠性;
- step3:计算参数匹配度；
- step4:通过考虑参数权重（attribute weight）和参数可靠性（attribute reliability），计算第k条规则的匹配度；
- step5:计算规则激活权重；
- step6:使用ER算法聚合激活的规则，得到最终的模型；
### 2.1、参数可靠性
&emsp;&emsp;假设前件的参数个数为M个，第i个属性的平均值为$\bar{x_i}$,第i个属性的标准差为$\sigma_i$，则该属性的公差范围为
$$\bar{x_i}-\psi\sigma_i<x_i<\bar{x_i}+\psi\sigma_i$$
其中，$\psi$是根据专家经验来指定的数值，用来调节公差范围的大小。当属性值$x^*\leq\bar{x_i}-\psi\sigma_i$或$x^*\geq\bar{x_i}+\psi\sigma_i$时，判断这个属性值是不可信的。
&emsp;&emsp;属性可信度计算流程如下，
![屏幕快照 20190914 下午1.50.32.png](0)
&emsp;&emsp;当第i个属性值$x_{ij}$不在公差范围内时，$y_{ij}=0$，反之，$y_{ij}=1$。对第i个属性的所有取值计算对应的$y_{ij}$，然后可以知道可信的属性值个数在全部属性值中占的比重$r_i$，这个$r_i$便是第i个参数的属性可信度。
### 2.2、匹配度
&emsp;&emsp;采用BRB模型的方法计算前件属性的匹配度,
$$\alpha_i^j = \begin{cases} \frac{x_{i(k+1)} - x_i^*}{x_{i(k+1)} - x_{ik}},&\ j=k,\text{if }x_{ik} \leq x_i^* \leq x_{i(k+1)} \\ 1 - \frac{x_{i(k+1)} - x_i^*}{x_{i(k+1)} - x_{ik}}, & j=k+1 \\0,& j=1,2,\ldots,|x_i^*|,j \not =k, k+1 \end{cases}$$
&emsp;&emsp;综合考虑属性权重和属性可信度，计算
$$\begin{aligned}
&C_i = \frac{\bar{\delta_i}}{1+\bar{\delta_i}-r_i} \\
&\bar{\delta_i} = \frac{\delta_i}{max_i=1,\ldots,T_k\{\delta_i\}}
\end{aligned}
$$
&emsp;&emsp;其中，$r_i$和$\delta_i$分别是第i个属性的可信度和权重。当属性i是完全可信的，即$r_i = 1$时，则$C_i=1$，否则，$C_i < 1$。
&emsp;&emsp;第i条规则的匹配度为
$$\alpha_k = \prod_{i=1}^{T_k}(\alpha_k^i)^{C_i}$$
&emsp;&emsp;其中，$T_k$为规则个数。
### 2.3、规则激活
