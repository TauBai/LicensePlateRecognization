# LicensePlateRecognization

车牌识别.


关于车牌定位的论文    http://blog.csdn.net/traumland/article/details/51204344


关于车牌的定位, 还是不知道能用什么办法能达到:

    随便输入一张包含车牌的图片, 将车牌定位并分割出来.  这样的效果 
    
        所有参数都是仅适用在某一确定场景
  
可能要像知乎中的一个回答所提到的 https://www.zhihu.com/question/46711294 , 要用到机器学习的方法, 

然而我这几天才刚刚接触这个, 刚开始看Andrew 的机器学习课程, 感觉还有很长一段路要走


  虽然知道再修改参数也是徒劳, 但一直徘徊不愿舍弃之前的方法. 有很长一段时间试图寻找普适的方法及参数, 未果, 遂暂放弃.
  
知乎@谢贤海 提到的openalpr我也看了它的部分功能, 第一次看开源软件有点找不到北, 好像分割用的mask是由adaboost提供的?

至于@piao lin 提到的eazypr, 我只看了相关的博客, 感觉和mastering opencv 这本书上的内容很相似


关于字符分割的内容    http://blog.csdn.net/traumland/article/details/51560135


关于字符分割, 传统方法上难点在怎么去噪怎么切割怎么去粘连, 怎么解决? 有人提到深度学习可以直接识别不用切割

关于字符识别, 虽说opencv有现成的程序, 但是不懂原理真的大丈夫?


到这里感觉好像机器学习成了万金油, 现在的人都在玩它. 之前的印象一直都是它的准确率不是很高, 大学期间没怎么接触过就把它放下了, 有些后悔. 


-------------------------------------
关于这个程序, 额..一开始我没有设计过自己的类, 照着c++ primer 讲的sales_data 弄得, 把所有程序都弄到了一个类中, 现在来看堆在一起实在太乱了. 中间几次想把它按功能分类, 但是有一些互相调用的地方, 想想就算了.

关于定位: 这个程序是利用了颜色与垂直边缘 取交集定位的, 方法有点老, 关于参数还是应该对应某特定场景去设置, 我这里没数据所以就不弄这里了

关于分割: 还是用了颜色信息进行分割, 因为效果比边缘要好很多

关于矫正: 用了霍夫变换(或ransac)检测车牌边框,用得到的四个点进行透视变换

关于字符分割: 用了投影分割法, 没用boundingbox, 他们两个不能说哪个好, 应该说都不怎么样, 看什么场景了

关于字符识别: 用了svm + 投影 + 像素信息 

关于这个程序目前有两点要解决

    最重要的是字符的特征选取, 我想要的尺度无关性特征, Mastering Opencv 提到的投影特征和像素特征肯定不行,
    
    我在犹豫用hog还是mser, hog与mser opencv都内置了, 但是我想先看看论文, 先看看别人是怎么做的, 目前也在找工作,有时间再弄吧
    
    还有一个是怎样在类文件里只需让svm读取xml一次? 
    没办法定义成全局变量, 在类构造函数中初始化也会出现这个错误 :error: undefined reference to `cv::ml::SVM::create(),
    且QT直接指向了这个函数      Ptr<_Tp> obj = _Tp::create();
                                                                                  2016.6.18
--------------------------------------------------------------------------------------------------------------------
后记:  关于这个程序(可以称得上是项目吗?)我弄了差不多一个多月, 说起来很容易也没多少内容, 如果现在要我做成这样的话, 两天就差不多够了, 甚至赶工一天就可能出来. 但要想知道每步为什么要这么做, 怎么做才好, 我觉得还是得多动手, 多查论文, 多实现. 要是照着Mastering Opencv这本书去扒, 是能很快做完, 但是会有你自己的思考吗? 比如说为什么选择vertical edge? 为什么那些论文说用vertical edge? 为什么要用sobel 不用其他edge detector? 照着人家做很快就做完, 会有一种我明白了的错觉,嘴上说,理论上讲, 而不实际去做还是不太好, 有的时候试一下可能就和你想的不一样. 
