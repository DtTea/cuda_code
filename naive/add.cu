#include <cuda.h>
#include <stdio.h>
// #include <sys/time.h>


#include < time.h >
#include <windows.h> //I've ommited this line.
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
struct timezone 
{
  int  tz_minuteswest; /* minutes W of Greenwich */
  int  tz_dsttime;     /* type of dst correction */
};
 
int gettimeofday(struct timeval *tv, struct timezone *tz)
{
  FILETIME ft;
  unsigned __int64 tmpres = 0;
  static int tzflag;
 
  if (NULL != tv)
  {
    GetSystemTimeAsFileTime(&ft);
 
    tmpres |= ft.dwHighDateTime;
    tmpres <<= 32;
    tmpres |= ft.dwLowDateTime;
 
    /*converting file time to unix epoch*/
    tmpres -= DELTA_EPOCH_IN_MICROSECS; 
    tmpres /= 10;  /*convert into microseconds*/
    tv->tv_sec = (long)(tmpres / 1000000UL);
    tv->tv_usec = (long)(tmpres % 1000000UL);
  }
 
  if (NULL != tz)
  {
    if (!tzflag)
    {
      _tzset();
      tzflag++;
    }
    tz->tz_minuteswest = _timezone / 60;
    tz->tz_dsttime = _daylight;
  }
 
  return 0;
}



double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void initCpu(float *hostA, float *hostB, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostA[i] = 1;
        hostB[i] = 1;
    }
}
void addCpu(float *hostA, float *hostB, float *hostC, int n)
{
    for (int i = 0; i < n; i++)
    {
        hostC[i] = hostA[i] + hostB[i];
    }
}
__global__ void addKernel(float *deviceA, float *deviceB, float *deviceC, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x; // 计算全局索引
    if (index < n)
    {
        deviceC[index] = deviceA[index] + deviceB[index];
    }
}
int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int n = 102400;

    hostA = (float *)malloc(n * sizeof(float));
    hostB = (float *)malloc(n * sizeof(float));
    hostC = (float *)malloc(n * sizeof(float));
    serialC = (float *)malloc(n * sizeof(float));
    initCpu(hostA, hostB, n);
    double stC, elaC;
    stC = get_walltime();
    addCpu(hostA, hostB, serialC, n);
    elaC = get_walltime() - stC;
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, n * sizeof(float));
    cudaMalloc((void **)&dB, n * sizeof(float));
    cudaMalloc((void **)&dC, n * sizeof(float));

    cudaMemcpy(dA, hostA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    int BLOCK_DIM = 1024;
    int num_block_x = n / BLOCK_DIM;
    int num_block_y = 1;
    dim3 grid_dim(num_block_x, num_block_y, 1);
    dim3 block_dim(BLOCK_DIM, 1, 1);
    addKernel<<<grid_dim, block_dim>>>(dA, dB, dC, n);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaMemcpy(hostC, dC, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    ela = get_walltime() - st;
    printf("n = %d: \n CPU use time:%.4f\n GPU use time:%.4f\n kernel time:%.4f\n", n, elaC, ela, ker_time / 1000.0);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}