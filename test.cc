#include <stdio.h>

int main(int argc, char const *argv[])
{
  unsigned char a, b;
  a = 100;
  b = 0.1f * a;
  printf("%d", a);
  printf("%d", b);
  return 0;
}
