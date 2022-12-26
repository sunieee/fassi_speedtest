import faiss
import numpy as np
from time import time

import numpy as np
d = 512                                           # 向量维度
nb = 1000000                                      # index向量库的数据量
nq = 100                                       # 待检索query的数目
np.random.seed(1234)             
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.                # index向量库的向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.                # 待检索的query向量


quantizer = faiss.IndexFlatL2(d)
nlist = 100
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

gpu_index = faiss.index_cpu_to_all_gpus(index)
t = time()
print(gpu_index.is_trained)
gpu_index.train(xb)
print(gpu_index.is_trained)
print('train time:', time() - t)

t = time()
gpu_index.add(xb)
print('add time:', time() - t)

gpu_index.nprobe = 10

t = time()
D, gt_nns = gpu_index.search(xq, 1)
print('inference time:', time() - t)

print(D.tolist())
print(gt_nns.tolist())
'''
3090双卡GPU：
从300w检索100个： 
train time: 0.9032609462738037
add time: 2.999061107635498
inference time: 0.018329143524169922
[[73.52603912353516], [70.35846710205078], [73.80403900146484], [72.24375915527344], [74.17415618896484], [71.22933959960938], [75.20762634277344], [72.21685791015625], [72.71469116210938], [70.68576049804688], [73.07495880126953], [75.25175476074219], [72.25598907470703], [72.27098083496094], [68.88560485839844], [70.9771957397461], [71.58976745605469], [74.25520324707031], [71.07829284667969], [71.30859375], [73.66899871826172], [73.4614028930664], [71.5515365600586], [67.87744903564453], [73.33705139160156], [73.13848876953125], [72.2054443359375], [71.57791900634766], [75.56784057617188], [68.38800811767578], [73.9201431274414], [72.62153625488281], [76.02925109863281], [69.12372589111328], [71.77532958984375], [76.25463104248047], [70.90967559814453], [71.6804428100586], [73.8437728881836], [72.86681365966797], [71.59480285644531], [71.18243408203125], [71.01715087890625], [73.47117614746094], [72.46540069580078], [73.05240631103516], [70.15145111083984], [74.34957885742188], [71.09028625488281], [72.77151489257812], [71.69477081298828], [73.30236053466797], [74.63739013671875], [72.5868148803711], [72.20756530761719], [71.21774291992188], [69.01313018798828], [71.16043853759766], [70.50656127929688], [73.17607879638672], [70.13896942138672], [72.1460952758789], [73.89676666259766], [70.5913314819336], [75.23268127441406], [70.0886001586914], [72.24005126953125], [73.53771209716797], [70.82806396484375], [69.08899688720703], [74.03872680664062], [70.35595703125], [73.0608139038086], [72.88457489013672], [71.2566146850586], [71.39507293701172], [77.33344268798828], [69.9053955078125], [70.66207885742188], [70.52965545654297], [73.94884490966797], [70.89154052734375], [71.92776489257812], [73.20114135742188], [74.10070037841797], [71.40804290771484], [68.61539459228516], [71.69819641113281], [73.20201110839844], [73.9726791381836], [71.86185455322266], [70.63458251953125], [67.62042999267578], [72.04788208007812], [71.46414184570312], [72.83590698242188], [72.64372253417969], [73.70257568359375], [68.37932586669922], [72.41232299804688]]
[[108], [1237], [720], [890], [259], [337], [520], [997], [427], [504], [210], [718], [479], [145], [1158], [1312], [81], [375], [369], [237], [811], [63], [412], [114], [142], [244], [128], [1995], [816], [792], [996], [111], [789], [1490], [38], [1349], [281], [642], [706], [1613], [680], [1289], [1374], [378], [1987], [567], [96], [1244], [885], [75], [300], [1145], [270], [824], [502], [553], [487], [551], [531], [1552], [1243], [132], [186], [378], [753], [765], [1969], [327], [350], [237], [909], [637], [237], [995], [141], [157], [1731], [401], [408], [426], [1256], [223], [525], [178], [300], [2041], [375], [1221], [1495], [1124], [3001], [33], [528], [739], [950], [369], [1906], [162], [1291], [670]]
'''