# PyTorch DeepSpeed Config JSON Documentation

## REQUIRED DeepSpeed Config JSON Parameters

***train\_batch\_size***: [integer]

| Value                                                        | Example |
| ------------------------------------------------------------ | ------- |
| The effective training batch size. This is the amount of data samples that leads to one step of model update. ***train\_batch\_size*** is aggregated by the batch size that a single GPU processes in one forward/backward pass (a.k.a., ***train\_step\_batch\_size***),  the gradient accumulation steps (a.k.a., ***gradient\_accumulation\_steps***), and the number of GPUs. | `32`      |

## OPTIONAL DeepSpeed Config JSON Parameters

### Batch Size Related Parameters

***train\_micro\_batch\_size\_per\_gpu***: [integer]

| Description                                                  | Default                      |
| ------------------------------------------------------------ | ---------------------------- |
| Batch size to be processed by one GPU in one step (without gradient accumulation). When specified, ***gradient\_accumulation\_steps*** is automatically calculated using ***train\_batch\_size*** and number of GPUs. Should not be concurrently specified with ***gradient\_accumulation\_steps*** in the configuration JSON. | ***train\_batch\_size*** value |

***gradient\_accumulation\_steps***: [integer]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Number of training steps to accumulate gradients before averaging and applying them. This feature is sometimes useful to improve scalability since it results in less frequent communication of gradients between steps. Another impact of this feature is the ability to train with larger batch sizes per GPU. When specified, ***train\_step\_batch\_size*** is automatically calculated using ***train\_batch\_size*** and number of GPUs. Should not be concurrently specified with ***train\_step\_batch\_size*** in the configuration JSON. | `1`       |



### Optimizer Parameters

***optimizer***: [dictionary]

| Fields | Value                                                        | Example                        |
| ------ | ------------------------------------------------------------ | ------------------------------ |
| type   | The optimizer name. DeepSpeed natively supports Adam and LAMB optimizers and will import other optimizers from [torch](https://pytorch.org/docs/stable/optim.html). | `"Adam"`                         |
| params | Dictionary of parameters to instantiate optimizer. The parameter names must match the optimizer constructor signature (e.g., for [Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)). | `{"lr": 0.001, "eps": 1e-8}` |

  Example of ***optimizer***

```json
"optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.001,
      "betas": [
        0.8,
        0.999
      ],
      "eps": 1e-8,
      "weight_decay": 3e-7
    }
  }
```

### Scheduler Parameters

***scheduler***: [dictionary]

| Fields | Value                                                        | Example                        |
| ------ | ------------------------------------------------------------ | ------------------------------ |
| type   | The scheduler name. See here for list of support schedulers. | `"1Cycle"`                      |
| params | Dictionary of parameters to instantiate scheduler. The parameter names should match scheduler constructor signature. | `{"lr": 0.001, "eps": 1e-8}` |

Example of ***scheduler***

```json
 "scheduler": {
      "type": "WarmupLR",
      "params": {
          "warmup_min_lr": 0,
          "warmup_max_lr": 0.001,
          "warmup_num_steps": 1000
      }
  }  
```

### Communication options

***fp32\_allreduce***: [boolean]

| Description                          | Default |
| ------------------------------------ | ------- |
| During gradient averaging perform allreduce with 32 bit values | `false`   |

***disable\_allgather***: [boolean]

| Description                  | Default |
| ---------------------------- | ------- |
| Disable allgather when using ZeRO optimizer and instead use broadcast | `false`  

***prescale\_gradients***: [boolean]

| Description                            | Default |
| -------------------------------------- | ------- |
| Scale gradients before doing allreduce | `false`   |

***sparse\_gradients***: [boolean]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Enable sparse compression of [torch.nn.Embedding](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) gradients. | `false`    |


### FP16 training options

***zero\_optimization***: [boolean]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Enable ZeRO memory optimization wrapper for FP16 Training. Currently compatible only with Adam optimizer. | `false`   |

***fp16***: [dictionary]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Configuration for using mixed precision/FP16 training that leverages [NVIDIA's Apex package](https://nvidia.github.io/apex/). An example, including the available dictionary keys is illustrated below. | None    |

```json
"fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
 	"min_loss_scale": 1
}
```

***fp16:enabled***: [boolean]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***enabled*** is a **fp16** parameter indicating whether or not FP16 training enabled. | `false`   |

***fp16:loss\_scale***: [float]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***loss\_scale*** is a ***fp16*** parameter representing the loss scaling value for FP16 training. The default value of 0.0 results in dynamic loss scaling, otherwise the value will be used for static fixed loss scaling. | `0.0`     |

***fp16:initial\_scale\_power***: [integer]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***initial\_loss\_scale\_power*** is a **fp16** parameter representing the power of the initial dynamic loss scale value. The actual loss scale is computed as 2<sup>***initial\_loss\_scale\_power***</sup>. | `32`      |

***fp16:loss\_scale\_window***: [integer]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***loss\_scale\_window*** is a **fp16** parameter representing the window over which to raise/lower the dynamic loss scale value. | `1000`    |

***fp16:hysteresis***: [integer]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***hysteresis*** is a **fp16** parameter representing the delay shift in dynamic loss scaling. | `2`       |

***fp16:min\_loss\_scale***: [integer]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| ***min\_loss\_scale*** is  a **fp16** parameter representing the minimum dynamic loss scale value. | `1000`    |

### Gradient Clipping

***gradient\_clipping***: [float]

| Description                         | Default |
| ----------------------------------- | ------- |
| Enable gradient clipping with value | `0`      |

### Logging

***steps\_per\_print***: [integer]

| Description | Default |
| ----------- | ------- |
| Print train loss every N steps | `10` |

***wall\_clock\_breakdown***: [boolean]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Enable timing of the latency of forward/backward/update training phases | `false`   |

***dump_state***: [boolean]

| Description                                                  | Default |
| ------------------------------------------------------------ | ------- |
| Print out state information of DeepSpeed object after initialization | `false`   |
