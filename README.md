# 主要参数
mnist lr:0.001

cifar lr:0.01
##聚合方法：s = fedAvg、trim、fltrust:
"server_num" : 1,
"server_agg": s
"rsu_agg" : "fedAvg",

##聚合方法：s = rohfl、our:
"server_num" : 4
"server_agg": s
"rsu_agg" : s


##攻击方法：
labelflip："evil_client_num" : 2 or 10
MPAF: "evil_rsu_num" : 3 * "server_num",
## lie攻击（rsu）
"server_num" : 1,
"rsu_num" : 1,
"client_num" : 100,
mnist "lr" : 0.007,
```python
{
    ...,
    prop : 0.8
}
```



```
python main.py -c ./utils/conf.json
```



## 15.5.5 效果对比

我们来看经过参数稀疏化之后，模型的性能表现如下图所示，可以看到，随着掩码矩阵中0的数量越来越多，稀疏化的模型性能在开始迭代时会有所下降，但随着迭代的进行，模型的性能会逐步恢复到正常状态。

<div align=center>
<img width="1200" src="./figures/fig33.png" alt="稀疏化效果表现"/>
</div>