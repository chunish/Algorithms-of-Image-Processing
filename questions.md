## strcpy函数的编写

```c
char *strcpy(char* dst,const char* src)
{
  assert((dst!==NULL)&&(src!=NULL));
  char* res=dst;
  
  while((*dst++=*res++)!='\0');
  return res;
}

```

## string 类的编写


## 构造函数与析构函数

> 构造函数：

1. 与类同名，进行初始化工作
2. 一个类中可以有多个构造函数，
3. **不能制定返回类型（包括）** 
