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

### string 类的编写
