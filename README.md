# CPM-1-Finetune-Text-Generation
**CPM (Chinese Pretrained Models)** 是北京智源人工智能研究院和清华大学研究团队合作开展的大规模预训练模型开源计划，清源计划是以中文为核心的大规模预训练模型。此项目微调了**CPM-LM**(2.6B)模型，用于**短文本生成**和**长文本生成**，具体任务对应于**歌词**和**小说**。
## Template
为了不让生成结果碰到\<eod>token导致生成其他任务类型的结果，使其生成结果都为**歌词**或者**小说**，所以采用如下模板：
### 歌词
    文本内容<eod>文本内容<eod>...文本内容<eod>文本内容<eod>
### 小说
    文本内容
## Result
### 歌词
数据集大小为**17.4MB**，使用**fp32**精度微调，微调了**15**轮，输出长度为**1024**。测试了**8**种不同风格的歌词，分别为**浪漫**、**伤感**、**治愈**、**怀旧**、**说唱**、**古风**、**快乐**、**思念**。从测试结果来看，**第10轮**的效果较好。8个结果放置于**results/lyric_fp32_27layer_eod_eod_10epoch_best**目录下。展示其中**两种**效果：

**浪漫**：

    Context(浪漫): 
    My Life 一直在等待<eod>
    空荡的口袋<eod>
    想在里面放 一份爱<eod>
    Why 总是被打败<eod>

    CPM: 
    我像是个回马枪
    插进你的世界里
    现在是凌晨三点半
    独自一个人在空荡的房间
    房间里传来disco音乐
    The only thing I wanna go now is drumming with you
    The laughter is jumping jack like a magnet my favorite skin
    I will drop it if you wanna ride with me
    把选择权还给自己
    无论结果是什么
    我都等着你 yeah
    Why 总是被击败
    我像是个恶梦骑士
    剌伤了你的心 还笑着说爱她
    我会配上不同颜色的子弹
    稀有的紫罗兰
    在我心中种下了 一起长大的祸祟
    那些过去的 绝不再回去重演
    MTFKRorevery
    我和你一模一样
    顾虑我的感受吗
    Why 总是被击败
    我像是个恶梦骑士
    刺伤了你的心 还笑着说爱她
    我会配上不同颜色的子弹
    稀有的紫罗兰
    在我心中种下了 一起长大的祸祟
    那些过去的 绝不再回去重演
    MTFKRorevery
    我和你一模一样
    顾虑我的感受吗
    Why 总是被击败
    别想躲着我了
    Id be your only one one
    别怕nobody love you
    受了伤 okay no way
    心中期待着你吧
    Yeah I wanna forever be your only one
    Put it on my own 就在我手里
    2020你来给我通风报信
    我像是铁路工人在挖着最深的隧道
    距离你爱上我还有1 yeah
    do you know what I mean
    do you know what I mean
    do you know what I mean
    我和你一模一样
    独自一人在空荡的房间
    房间里传来disco音乐
    the only thing I wanna go now is drumming with you
    The laughter is jumping jack like a magnet my favorite skin
    I will drop it if you wanna ride with me
    把选择权还给自己
    无论结果是什么
    我都等着你 yeah
    Why 总是被击败
    我像是个恶梦骑士
    刺伤了你的心 还笑着说爱她
    我会配上不同颜色的子弹
    稀有的紫罗兰
    在我心中种下了 一起长大的祸祟
    那些过去的 绝不再回去重演
    MTFKRorevery
    我和你一模一样
    顾虑我的感受吗
    Why 总是被击败
    我像是个恶梦骑士
    刺伤了你的心 还笑着说爱她
    我会配上不同颜色的子弹
    稀有的紫罗兰
    在我心中种下了 一起长大的祸祟
    那些过去的 绝不再回去重演
    MTFKRorevery
    我和你一模一样
    顾虑我的感受吗
    Why 总是被击败
    我像是个恶梦骑士
    刺伤了你的心 还笑着说爱她
    我会配上不同颜色的子弹
    稀有的紫罗兰
    在我心中种下了 一起长大的祸祟
    那些过去的 绝不再回去重演
    MTF[1024]
    
**说唱**：
    
    Context(说唱): 
    他们都懂得现实如同苦难戏<eod>
    他们都知道虚幻只出现在电视剧<eod>
    挥洒着金钱只是我的本性<eod>
    这就是为什么许多人都叫我mr King<eod>

    CPM: 
    你必须看着我
    我的生活你永远不能比拟
    用keep ur eyes visions 的视眼射击全世界
    你必须看着我
    每个进入我的视野的女子
    都会成为我的邻家女孩
    玛莎拉蒂和louis gucci在房间里
    我点了一杯whisky 酸莫高窟
    在这个迷人的夜晚 我和我的兄弟都是属于我的
    你必须看着我
    无法得到理解的孤独眼神
    那些假装的不被打击到的脆弱
    只是为了他们自己 没有人能和我比
    他们都懂得现实如同苦难戏
    他们都知道虚幻只出现在电视剧
    挥洒着金钱只是我的本性
    这就是为什么许多人都叫我mr King
    你必须看着我
    我的生活你永远不能比拟
    用keep ur eyes visions 的视眼射击全世界
    你必须看着我
    每个进入我的视野的女子
    都会成为我的邻家女孩
    玛莎拉蒂和louis gucci在房间里
    我点了一杯whisky 酸莫高窟
    在这个迷人的夜晚 我和我的兄弟都是属于我的
    你必须看着我
    无法得到理解的孤独眼神
    那些假装的不被打击到的脆弱
    只是为了他们自己 没有人能和我比
    他们都懂得现实如同苦难戏
    他们都知道虚幻只出现在电视剧
    挥洒着金钱只是我的本性
    这就是为什么许多人都叫我mr King
    你必须看着我
    我的生活你永远不能比拟
    用keep ur eyes visions 的视眼射击全世界
    你必须看着我
    每个进入我的视野的女子
    都会成为我的邻家女孩
    玛莎拉蒂和louis gucci在房间里
    我点了一杯whisky 酸莫高窟
    在这个迷人的夜晚 我和我的兄弟都是属于我的
    你必须看着我
    无法得到理解的孤独眼神
    那些假装的不被打击到的脆弱
    只是为了他们自己 没有人能和我比
    他们都懂得现实如同苦难戏
    他们都知道虚幻只出现在电视剧
    挥洒着金钱只是我的本性
    这就是为什么许多人都叫我mr King
    你必须看着我
    我的生活你永远不能比拟
    用keep ur eyes visions 的视眼射击全世界
    你必须看着我
    每个进入我的视野的女子
    都会成为我的邻家女孩
    玛莎拉蒂和louis gucci在房间里
    我点了一杯whisky 酸莫高窟
    在这个迷人的夜晚 我和我的兄弟都是属于我的
    你必须看着我
    无法得到理解的孤独眼神
    那些假装的不被打击到的脆弱
    只是为了他们自己
    没有人能和我比
    他们都懂得现实如同苦难戏
    他们都知道虚幻只出现在电视剧
    挥洒着金钱只是我的本性
    这就是为什么许多人都叫我mr King
    你必须看着我
    我的生活你永远不能比拟
    用keep ur eyes visions 的视眼射击全世界
    你必须看着我
    每个进入我的视野的女子
    都会成为我的邻家女孩
    玛莎拉蒂和louis gucci在房间里
    我点了一杯whisky 酸莫高窟
    在这个迷人的夜晚 我和我的兄弟都是属于我的
    你必须看着我
    无法得到理解的孤独眼神
    那些假装的不被打击到的脆弱
    只是为了他们自己
    没有人能和我比
    他们都懂得现实如同苦难戏
    他们都懂得虚幻只出现在电视剧
    挥洒着金钱只是我的本性[1024]
### 科幻小说
小说数据集都为**科幻**类型的，大小为**8.4MB**，使用**fp32**精度微调，微调了**10**轮，输出长度为**2048**(使用了自己编写的**无限生成**脚本)。测试了**4**种不同科幻小说，分别为**吞噬地球**、**吞噬星空**、**时间机器**、**白垩纪往事**。从测试结果来看，第**6**轮的**科幻内容**生成效果较好。5个结果放置于**results/novel_fp32_27layer_no_eod_6epoch_best**目录下。展示其中**两种**效果：

**吞噬地球**：

    Context(吞噬地球): 作为联合国地球防护部队在太空中的最高指挥官，他率领的这支小小的太空军队装备着人类有史以来当量最大的热核武器，敌人却是太空中没有生命的大石块。
    在预警系统发现有威胁地球安全的陨石和小行星时，他的部队负责使其改变轨道或摧毁它们。这支部队在太空中巡逻了二十多年，从来没有一次使用这些核弹的机会。那些足够大的太空石块似乎都躲着地球走，故意不给他们创造辉煌的机会。

    CPM: 这天,马宝瑞又拿起了那张世界地图,那是用六张海地图拼成的,他不动声色地划着,像是变魔术似的,转眼间那张大图变成了一张嵌入到大海中的地图,海洋从地图上看也变成了大海,地图上的石块像是从一只大的平放的碗中随意掉落到碗中的,没砸到女青年,但砸到没举手的游客们了。他厉声说:“快把这张桌子搬走!”没人说话,这个森严的太空阵地又报警了,这次响是接着那个霹雳舞,只见一大片蜈蚣似的电磁动能电弧不断从太空部队分布在各个位置,不断被激发,那些抛出的石块击中了小镇的每一个物体,却不伤及水中的人体,引来了一群群狂欢的蜈蚣。“镇政府藏到哪儿了?”马宝瑞又问一名上尉。“距此不到五公里,镇长是个老瘪三,”上尉说,“已经让学潮耽误了,下次会还会这样。”“下次?再下次是多少年以后?”“也许是两万年之后,也许是一万年以后吧。”这时,预警系统报警,预警中心上空出现了一个高达二十米的棒状体,看上去像是一个泄落的太阳,“恒星型氢弹”很快出现了,在十几公里外,马宝瑞就穿着一身银色的战术服从这个氢弹中走了出来,他没戴手套,手起刀落,一个也连不着可燃的肢体被齐胸砍成两半,一大群活着的蜈蚣在雪地上乱窜,每个吓跑十到十五条,但也很快被他追上了。那个氢弹在空中发出了相当响亮的爆炸声,小镇上空当真像是开了锅,蘑菇状的氢弹气焰逼人,吓得游人纷纷从不同的方向逃命,整个世界,变成了一个氢弹爆炸的太空遥远世界,仿佛在刚刚过去的那场战争中仍在发生。几十名太空特警分别戴在两肩上,作为士兵分布在全世界的各个位置,这时全世界都处于联合国和舰队的统一指挥下,用起来更像一个战争指挥组。那些太空部队迅速控制了局面,上至权倾一时的首相,下至幼稚园的阿姨,都自觉地搬开了一个个的战略物资,成为买通的游客们争相观看的对象。只有这时,马宝瑞才想起了她,重新戴好手套,拿起一副橡木医用手套,穿上红外假发,假发漂浮在那片火焰般的大石头后面,接触大自然的火焰让马宝瑞在心中对一种叫“氢弹”的太空武器有了朦胧的概念。他拉动假发的两个帽沿,把迎面飞来的几粒石子、一根从假发帽沿上掉落的胡萝卜和一个砍下的鸡脚生绑到了假发上。假发后面的太空防御系统读到了这些信息,发出了警报,过去不明真相,纷纷效仿,也不知马宝瑞是否想过要为地球文明保存几块这样的石头和鸡脚,现在都有人佩戴,并把它们与阿尔法假面具连在一起,以此表达对太空战略武器的敬畏。马宝瑞戴上假发,对着蹲在小镇街头的一百九十八个人挥了挥说:“你们呢,赞成拆除武器、生化炸弹和工兵连的人呢?”没有人回答,这些人这时都跑到比萨斜塔下去仰望太空了。马宝瑞又挥了挥手,那些太空特警们摘下假发,又对着空中举着假发的人一一叫出了他们的名字,有许多人还是第一次听到有人叫[1024]自己,都在纳闷地看着他。那些举着假发的人都跑下了斜塔,就在推向太空的尸水与稍后推送过来的尸水会合处,有一条绳梯从几十米的高度正降下来。“站住”马宝瑞大喊一声,那些尸水吧啦吧啦地全部悬停在尸水之上,只有在缓缓下落的尸水和上升的尸水这一动态的特征,没办法识别出尸体是男是女。有婴儿肥的婴儿脸蛋被尸水带着一起下降,咔嚓、咔嚓,好听的声音从高处传来,婴儿肥女人的双腿则像悬空一样,美得让人都忘记了抱住她。下而下,又有七八个人这样仰望着太空被尸水带到地面上,分别是婴儿的姥姥、老农、黑瘦的男人、白衬衣的男人、小鲜肉,还有一群男女生剪在一起的好像是在说“我会舞会弹钢琴”,双方的尸水此时都结了厚厚实实的一层,谁都不知道谁是谁,而这就对演习内容提供了一个大好素材。这一切结束得太快,刚刚天黑,警报就响彻天空。马宝瑞听到话机中的声音抖了一下:“我是基地最高指挥官,你、你在哪里?”“我在大广场,马上到广场来,尽快,否则后果自负。”马宝瑞觉得声音发抖,他仔细听着:“快,就现在!!”“哒哒哒——”话机里的声音衰落了,显然是对基地外的远程武器警告声进行了屏蔽。其实这时即便用上声音也分辨不出来,在这瞬息万变的太空何来禁声一说?但马宝瑞显然从那个声音上得到了些什么,脸色煞白,用另一只袖子掩住了嘴,只是喃喃自语:“哒哒哒——”这时候,几分钟前俯冲到地面的尸水又哗哗地沸腾起来,此起彼伏地响着,话机、假发、和尸水都颤动着,生怕瞬息刹住尸水的下势,紧接着有嘶哑的声音响起:“哒哒哒——”“怎么回事?”太空指挥官诧异地问,“是尸水被屏蔽了,还是尸水对你的话筒了?”“我的话筒!我的话筒!”马宝瑞绝望地大叫起来,“快,把我的话筒拿来,我要和话筒**!”“什么?”声音突变,竟然是娇俏可人的声音,是学院派的那种娇俏:“古娜拉黑暗吧,大神你在哪里?我的法术都做好了,我向那里超级喷射!”“超级喷射?”马宝瑞觉得天旋地转,“那里是深海......多少倍音高?”“100万!”声音铿锵有力。马宝瑞失声问道:“100万倍音高?那是什么?声索吗?”“是啊,是超级超声波!”“那里有几十万、几百万倍音高的话筒,能听[2048]

**白垩纪往事**：

    4.Context(白垩纪往事): 卡奇卡说："我知道一些：反物质是恐龙物理学家们猜想中的一种物质，它的原子中的粒子电荷与我们世界中的物质相反。反物质一旦与我们世界的正物质相接触，双方的质量就全部转化为能量。"乔耶点点触须说："现在大家知道有比核武器更厉害的东西了，在同样的质量下，正反物质湮灭产生的能量要比核弹大几千倍！"

    CPM: "那就是说,恐龙们还未掌握制造反物质的技术?""这还未确定,但我想那种技术要等到他们研制出足够数量的产品,如流水线似的制造出金字塔那样高的反物质反应堆时才能实现。"于是,在这一天当中,世界发生了一件任何黄金都无法与之比拟的事:真理祭坛前的非洲方阵与日冕,在非洲大陆的中心恰恰发生湮灭,恰逢其时,没有几个国家可以抵御,最早抵抗的是美洲、次之是欧洲、然后是非洲,粤语大军就是在这最后的时刻,用了化电费甚么的,向真理祭坛上的发现了的反物质群发了一通火,又把自己日封面了一声哨,世界就在火海中了。这火不白野,因为这是真理祭坛上的反物质在加热后发生的核反应,任何物质都无法在核反应中存活,除了变成纯能量。所以,这火海中跑的都是高能耗的物质,耗油不多却很有效,每条跑进去几个人,就把那一片地区燃烧五分钟。这一来,引起了整个日冕的连锁反应,使地球上的气候倒退了五百年。这一来更麻烦,太阳被反物质高温烧熔了,但在另一方面,人进入了熔岩湖,高温汽化了他们的身体,使他们的身体进了动物界,成了全然不同的物种。最惨的是知识人类,因为反物质是非定制式武器,它们在加热时不稳定,必须藉仪器进行校正,但校正后的反物质不能再反应,也不能用於武器制造,只有把大多数人都打进大海里,才能在短期内把人类都海洋格式化,使人类文明彻底熄火。世界各国都对非洲及世界气象组织求救,但当时气象专家担心各国不管也不分,还怕非洲联合体从中作梗,实际上非洲己经联合,当非洲联合体成立时,它的气象观测基地已达到了京都标准,其轨道卫星、气球和浮力装置都能定量发射风切量,连官员都不相信这些装置会比气象卫星更精确。当风切克wei枪出现时,气象部只好把联合国气候变化专门委员会的张伯生喊回来,于七月七日升起了清道夫。张伯生是沿海海面上的一座灯塔,那些风切克wei枪一出现,它立刻放出了必要的蒸气给海洋,做了一些急事,所以气象厅升起这个灯塔有些讽刺的意思。简单地升起后,日冕越来越远,真理祭坛周围的人也越来越多,当升至天文台的高度时,大约达到了四千多人,这时,太阳已越过了西弦月,高温汽化了整个日球,真理祭坛前聚集了全世界的目光,更加剧了各国的忧心,气象卫星的实时实况不仅有行星运行的实况,还有极光的实况,所以当极光出现时,所有的国家都在实况上看到了极光的亮点。在这之前,各国都认为,制造超高速飞行器加上太阳极光,就能制造出日冕来。但那时非洲已经有了宇航员,所以把宇航员都捆到了祭坛上。当牧首升起太阳旗帜时,就从全球观测宇宙中,宇宙定位系统!这个系统使全球绝大部分位置都被设定在一个天文单位,但每次打击目标时位置必须在这个规定的位置以外,就是说,要在打击中至少在白天才能打击到目标。打击时生成的牺牲品一般是极光电台,因为要通过发射卫星或气球广播等方式把声音向整个宇宙传播,所以叫做信天翁电台,其实就是以前的气象录音机。打击时通讯路线只能是直线,因为从位置上讲,信天翁电台[1024]在西弦月出没的区域内就是一个死胡同,所以只能用一架飞行器飞向它们出没的区域,叫飞鹰。当以上三架飞行器全部飞行后,将只探明信天翁电台的方位,科学家们知道,那个方位就是地球三大构成群的引力平衡方位,现在,三个相互作用范围分别是四大洋、只有土星、只有水星,就是三大构成群在三大大洋中的聚集地,当然,这种宇宙定位只能达到三成就是可信赖的,即使考虑到早期各构成群的位置对信天翁电台的干扰,探明三大行星的引力平衡方位也需两万至三万年才能实现,但目前人类探知的四大洋中的土星、水星、地球和太阳都被方位设定了,估计很快就能定位完成,所以这是人类首次精确、直觉地定位太阳系的四大构成体。当然,这不是全球定位系统的开始,当定位完成后,全球将在几分钟内成为一个定位平衡的世界,所以那时光污染就必然成为问题,这可能是人类世界的第一次清晰明朗的天空。其实当时真正需要考虑的是太阳的方位,在太阳出现后,亮度决定于辐射,所以按照现在的方位所需要的量度,可能正好等于零。在日出后,极光出现的范围不断扩大,当达到麦肯齐晴天时,全世界都能看到,这时只有南极大陆上的人能够看到。当时,大部分人都认为除了恒纪元的高维生命能看到之外,现在可知宇宙九成的空间都能看到,但现在最多的估计是太阳方位能精确到一度。目前,非法地传送到地球的开阔天空的是极光十二次,这表明今后宇宙能传送的开阔天空可以达到十二小时,这给人类后来的家园开拓了许多可能。这天夜里,北半球无风。这时,日巡者一号定位系统的高度估计在四千米左右,对东西太阳全方位搜索,不管用,只搜索到东太阳一颗卫星。定位系统对它进行方位定位,要在三十二小时之内,所以东太阳那里启明星很多,但不在范围之内,启明星已经移到蛇人后方的天空了,启明星向来都是智慧体的标志,所以启明星不可能是太阳的方位。这提示了日巡者的第一个危险方位:东方。启明星不多但很亮,至少要斟酌整个方位,所以启明星应该是东方。但随着地位的接近,启明星在默契和认知上,以及启明星对太阳一致的启迪,启航者们渐渐忘记了启明星,进入了完全的盲目状态;而日巡者三号很快意识到了这一点,它把启明星从东方移到了运行目标区的正西方。这就产生了一个危险,随着启航者们对太阳方位的认知激增,东方应该出现危险信号,因为启明星往往出现在启明星原生长的方向(从启明星的外形就能看出来,这是一棵榕树),所以应该立刻修正过来。但日巡者三号知道,本人启明星的方位是不可能跳跃到目标区的西方的,所以启明星是东方也不保险。启明星不多但很亮,肯定是东方,这是定位启航者最初的直觉,以后随着启航者们地位的接近,这一直觉越来显露出来[2048]
