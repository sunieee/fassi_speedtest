# fassi测试

https://github.com/facebookresearch/faiss



### 使用多个标签

将单例改为多个标签一起inference

测试用例：28组promt，共703个词

|          | 单个标签 | 多个标签一起 |
| -------- | -------- | ------------ |
| Default  | 4.51746  | 3.82024      |
| Flat     | 5.48922  | 3.71875      |
| IVFxFlat | 3.8263   | 3.21174      |
| QPx      | 7.25442  | 3.5519       |
| IVFxPQy  | 3.18878  | 3.23568      |
| LSH      | 7.3807   | 3.6455       |
| HNSWx    | -        | -            |

HNSWx报错而无法运行，报错为：

```
==================== HNSWx ====================
train time: 0.0
Faiss assertion 'p + size == head_' failed in void faiss::gpu::StackDeviceMemory::Stack::returnAlloc(char*, size_t, cudaStream_t) at /root/miniconda3/conda-bld/faiss-pkg_1669821803039/work/faiss/gpu/utils/StackDeviceMemory.cpp:144
Aborted (core dumped)
```

Default加速比 = 4.51746 / 3.82024 = 1.18

### 统计

阈值为0.05

|          | inference time | train time | add time | acc     | #modify | #right |
| -------- | -------------- | ---------- | -------- | ------- | ------- | ------ |
| default  | 3.86058        | 0          | 0        | 1       | 703     | 703    |
| Flat     | 3.8599         | 0          | 1.01498  | 0.48649 | 345     | 342    |
| IVFxFlat | 3.32033        | 1.04713    | 1.15526  | 0       | 0       | 0      |
| PQx      | 3.65089        | 8.40904    | 3.05794  | 0.62731 | 703     | 441    |
| IVFxPQy  | 3.30469        | 4.54668    | 1.24999  | 0.71408 | 703     | 502    |
| LSH      | 3.70182        | 0          | 0.61448  | 0.52063 | 477     | 366    |

阈值为0

|          | inference time | train time | add time | acc     | #modify | #right |
| -------- | -------------- | ---------- | -------- | ------- | ------- | ------ |
| default  | 3.86128        | 0          | 0        | 1       | 703     | 703    |
| Flat     | 3.80253        | 0   | 1.00874  | 0.96017 | 678     | 675    |
| IVFxFlat | 3.28365        | 0.99777    | 1.14851  | 0       | 0       | 0      |
| PQx      | 3.61544        | 8.35615    | 2.98433  | 0.62731 | 703     | 441    |
| IVFxPQy  | 3.26414        | 4.35283    | 1.22339  | 0.70413 | 695     | 495    |
| LSH      | 3.7397         | 0          | 0.60988  | 0.52063 | 477     | 366    |


\#modify表示703个中修改过（匹配上）的单词，#right表示正确的单词

- 匹配数量（#modify）：发现使用default模型703个此都能匹配上，所以阈值thr并不重要。但是后面即使阈值取得非常低，部分方法#modify （修改的数量）也不多
- 准确率（acc=#right/703）：应该是越快acc越低，速度越慢(接近default)，acc就越高，但是这里都不太高。只有Flat再阈值为0时相当。
- 运行时间（inference time）：貌似并没有比default快多少呢，最快的方法（IVFxPQy）加速比为 3.86/3.32=1.16，甚至低于使用多个标签的加速比

