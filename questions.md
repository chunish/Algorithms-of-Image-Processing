## 1. strcpy函数的编写

```c
char *strcpy(char* dst,const char* src)
{
  assert((dst!==NULL)&&(src!=NULL));
  char* res=dst;
  
  while((*dst++=*res++)!='\0');
  return res;
}

```

## 2. string 类的编写



## 3. 构造函数与析构函数

> 构造函数：

* 与类同名，进行初始化工作
* 一个类中可以有多个构造函数，
* 不能制定返回类型（包括void）
* 不带参数的构造函数：只能以固定不变的值初始化对象
* 带参数的构造函数相对灵活

> 析构函数

* 离开工作域时调用的函数：完成清理工作，释放内存
* 名字同类名，但前加~
* 一个类只允许有一个析构函数
* 不能有参数，无返回值（包括void）


## 内存的分配：mallco, free, new, delete等的使用
用于申请动态内存和释放内存
> mallco/free
* 



## 枚举的使用：enum/typedef enum


