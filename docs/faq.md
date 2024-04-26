# FAQs

## What's this?

HGQ is a method for quantization aware training of neural works to be deployed on FPGAs, which allows for per-weight and per-activation bitwidth optimization.

## Why is it useful?

Depending on the specific [application](https://arxiv.org/abs/2006.10159), HGQ could achieve up to 20x resource reduction compared to the traditional `AutoQkeras` approach, while maintaining the same accuracy. For some more challenging [tasks](https://arxiv.org/abs/2202.04976), where the model is already under-fitted, HGQ could still improve the performance under the same on-board resource consumption. For more details, please refer to our paper (link coming not too soon).

## Can I use it?

The following conditions must be met:

1. Your model is competible with `hls4ml` (i.e. it can be converted to HLS C++ code).
2. You are using `Vivado` as your FPGA backend.
   - However, other backend MAY work if you don't use heterogeneous activation quantization.
3. You are using `tensorflow.keras` as your model API.
4. Your model is fully connected or convolutional.
   - i.e. no RNN, LSTM, etc. transformers should work if you build one with dense and the conv1d for MMM hack.
5. You are using `tensorflow` as your training framework, and you are using `tf.keras` as your model API.
   - Supports both `Sequential` and `Functional` keras model API.

If you meet all the above conditions, you can probably use HGQ to quantize your model.

## What's the status of the project?

The project is still under development. The codebase and documentation are not stable yet, and we are working on it. If you have any questions, please feel free to contact us.

## I'm getting terrible results during/after training, what should I do?

Do you observe a collapse of loss value, as well as BOPs and accuracy? If so, you may want to try the following:

1. Reduce `beta` in `H-` layers.
2. Reduce or remove the regularization quantizer config.
3. Increase `init_bw` in quantizer config.
4. If it still doesn't work, please first verify that the Keras counterpart of your model is working properly. If it is, please feel free to contact us.

## `ERROR: [XFORM 203-504] Stop unrolling loop 'SomeName'` for Vivado HLS

Your layer is too big. No, actually, due to some hard-coded and yet stupid heuristic in Vivado HLS, it stops you from unrolling some large and/or nested loops regardless of whether it is actually synthesizable/large (e.g., a large but very sparse dense layer will fail). There is **NO** way to disable this check. You can try to reduce the size of your layer, or try to use `parallel_factor` for convolutional layers, or split dense layers manually.
