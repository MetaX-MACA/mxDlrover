This is an example of colosalai distributed checkpoint in which only save and load distributed checkpoint.

## Requirements:
```
colossalai >= 0.3.6
```
The distrubed checkpoint for HybridParallelPlugin is supported since this version.


## checkpoint format:
The difference between dlrover and native checkpoint is only the **dlrover_latest.txt** is needed for flash-checkpoint in dlrover
```
/gpfs/userdata/m00915/colossalai_train/ckpt
├── [   1]  dlrover_latest.txt
├── [   1]  latest.txt
├── [4.0K]  step-1
│   ├── [ 761]  lr_scheduler
│   ├── [8.0K]  modeling
│   │   ├── [ 718]  config.json
│   │   ├── [ 111]  generation_config.json
│   │   ├── [ 29K]  pytorch_model.bin.index.json
│   │   ├── [1022M]  pytorch_model-stage-00001-shard-00001.bin
│   │   ├── [986M]  pytorch_model-stage-00001-shard-00002.bin
│   │   ├── [1008M]  pytorch_model-stage-00001-shard-00003.bin
│   │   ├── [1008M]  pytorch_model-stage-00001-shard-00004.bin
│   │   ├── [986M]  pytorch_model-stage-00001-shard-00005.bin
│   │   ├── [944M]  pytorch_model-stage-00001-shard-00006.bin
│   │   ├── [472M]  pytorch_model-stage-00001-shard-00007.bin
│   │   ├── [986M]  pytorch_model-stage-00002-shard-00001.bin
│   │   ├── [1008M]  pytorch_model-stage-00002-shard-00002.bin
│   │   ├── [1008M]  pytorch_model-stage-00002-shard-00003.bin
│   │   ├── [986M]  pytorch_model-stage-00002-shard-00004.bin
│   │   ├── [944M]  pytorch_model-stage-00002-shard-00005.bin
│   │   ├── [986M]  pytorch_model-stage-00002-shard-00006.bin
│   │   └── [508M]  pytorch_model-stage-00002-shard-00007.bin
│   ├── [4.0K]  optimizer
│   │   ├── [ 111]  pytorch_optim.bin.index.json
│   │   ├── [1.3K]  pytorch_optim_group.bin
│   │   ├── [ 555]  pytorch_optim-stage-00001-shard-00001.bin
│   │   └── [ 555]  pytorch_optim-stage-00002-shard-00001.bin
│   └── [  65]  running_states.json

```

## example:
you can simplely try distributed checkpoint  in k8s cluster by adaption the following case in which dlrover is deployed:
```
kubectl create -f llama70b_job.yaml
```
