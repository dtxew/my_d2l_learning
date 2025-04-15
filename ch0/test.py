# #字符串格式化
# print("%d-%02d" % (3,1))

# print("hello,{}:your age:{}".format("小明",23))


# #f-string:以f开头的字符串，字母会以变量替代
# r=2
# s=3.14*r*r
# print(f"圆的半径为{r},面积为{s:.2f}")

# #字典
# d={1:'a',2:'b',3:'c'}

# a=eval(input())

# #判断元素是否在表中
# if a not in d:
#     print("none")
# else:
#     print(d[a])

# #可变参数和关键字参数

# def sum(*array):
#     res=0
#     for i in array:
#         res+=i
#     return res

# print(sum(1,2,3,4,5,6,7,8,9,10))
# print(sum(1,2,3,4,5))

# def printkw(**kw):
#     print(kw)

# printkw(city="beijing",age=12)

# #只接收星号后面的关键字的函数
# def printInfo(name,age,*,city,job):
#     print(f"名字:{name},年龄:{age},城市:{city},工作:{job}")

# printInfo("zhangsan",22,city="beijing",job="none")

# #字典迭代

# d={1:'a',2:'b',3:'c'}

# for k in d:
#     print(k)

# for v in d.values():
#     print(v)

# for k,v in d.items():
#     print(f"{k}:{v}")

# #下标索引

# a=[1,2,3,4]

# for i,x in enumerate(a):
#     print(f"{i}:{x}")

# #列表生成式,if在for后面不能使用else但在前面

# a=[x*x for x in range(11) if x%2==0]
# print(a)

#生成器,把上面的[]改成()

# g=(x*x for x in range(11))

# print(next(g))
# print(next(g))

# for x in g:
#     print(x)


##函数生成器，生成完后从生成语句继续往下执行
# def odd():
#     yield 1
#     yield 3
#     yield 5

# o=odd()
# print(next(o))
# print(next(o))
# print(next(o))

# #迭代器

##高阶函数：函数也可以当作变量