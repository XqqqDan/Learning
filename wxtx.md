##### **小尺度衰落信道的分类**

-   信道衰落速率：慢:路径损耗(只与距离相关) 中:阴影衰落 快:多路径衰落
-   时变角度（多普勒频移）：快衰落和慢衰落

    -   快衰落
        -   频域：信道相干时间 < 信号周期
        -   时域：信号带宽 < 多普勒平移
    -   慢衰落
        -   频域：信号周期 < 信道相干时间
        -   时域：多普勒平移 < 信号带宽

-   多径角度：频率选择型衰落，平坦衰落

    -   平坦衰落
        -   频域：信号带宽 < 信道带宽
        -   时域：信道时延扩展 < 信号周期
    -   频率选择型衰落
        -   频域：信道带宽 < 信号带宽
        -   时域：信号周期 < 信道时延扩展

-   频率选择型衰落对信号的影响：会造成符号间干扰，信号重叠（码间干扰）
-   小尺度衰落在短距离时有变化较快

##### **自由空间传播模型**

-   应用：卫星通信
-   接收功率与距离平方成反比，与波长平方成正比（与频率平方成反比）
-   为什么大于某个距离之后功率会快速下降：两个电磁波出现反相情况，快速抵消

##### **多径信道模型**

-   接收端接受到信号会有一个时延
-   时延比较大的时候会出现频率选择型衰落
-   时域上出现符号干扰
-   为什么天线间的距离设置成半波长：距离为半波长的时候，天线之间的相关性比较小

##### **瑞利衰落**

-   瑞丽分布中信号无直射路径，通过反射衍射等方法到达接收端，莱斯分布中信号有直射路径
-   接收信号的实部和虚部服从高斯分布，包络服从瑞丽分布，功率（包络平方）服从指数分布

---

##### **信道容量**

-   CSIT 名词解释：发送端信道状态信息
-   哪些技术需要知道 CSIT：注水法，预编码，自适应调制，信道反转
-   Jensen 不等式的物理意义：在低信噪比情况下，注水法可以获得比高斯信道更好的容量
-   只有在时间上注水有概率项
-   用注水法求信道容量！

##### **数字调制**

![example1](/pics/%E5%9D%90%E6%A0%87%E4%BE%8B%E9%A2%98.png)

-   信座图上的点和原点距离的物理含义：功率的根值
-   16（任意）com 的信号要多少多少支路：2 路（即需要 2 个基向量来表示）
-   符号速率越高，容错越高，错误概率越低（由多普勒平移引起的）
-   多天线工作在分频模式，能获得分频增益吗？不能，因为天线是相关的，但可以获得阵列增益

![example2](/pics/%E4%BE%8B%E9%A2%982.png)

---

##### **分集&MMSE**

-   1.阵列增益：多天线即可，与是否独立无关 2.分集增益：独立才有，相关不行
-   分集的作用：用于对抗衰落
-   发送分集在什么时候可以获得和接收分集一样的性能：在知道发送端信道状态信息（CSIT）时
-   选择合并如何求第 i 条支路的平均信噪比：
    -   写概率密度函数
    -   求第 i 条支路的中断概率,有 m 条支路时要有 m 次方
-   在 MRC 中，发送端有 m 根天线，接收端 1 根，阵列增益和分集增益是多少：阵列是 m 倍，分集是 m 次方
-   什么技术可以获得香农容量，达到容量上界：MMSE-SIC
    ![三种合并方式](/pics/2pics.png)

##### **均衡&OFDM**

-   为什么要引入多载波技术：减少码间干扰
-   如何减少码间干扰：1.发送端使用预编码 2.接收端使用均衡器 3.使用多载波 OFDM 技术
-   发射功率增加可以减少码间干扰吗？不能，因为噪声也增加了
-   OFDM 如何缓解频率选择性衰落：将数据分到子载波传输，变成平坦信号
-   循环前缀的优缺点：
    -   优点：1.将线性卷积变为循环卷积，解决码间干扰 2.克服多径时延的影响 3.可以使均衡(接收机，解调)变简单:Y=A\*X
    -   缺点：引入了冗余信息，降低了频谱效应

##### **MIMO**

![MIMO](/pics/MOMO.png)

-   4 X 4 的 MIMO 系统的复用增益为 2，分集增益是多少：d=(4-2+1)X(4-2+1)=9

##### **其他**

-   误比特率越高，频谱效应越高，容错率越高
-   在通信系统中，有效性和可靠性是矛盾的，请举例：
    -   分集和复用：分集提高可靠性，复用提高有效性
    -   调制：二进制调制和多进制调制
    -   信道编码：增加冗余比特（可靠性增加，有效性减少）
