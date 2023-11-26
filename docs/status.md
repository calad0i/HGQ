# Status

HGQ is still under development. The following table shows the current status of the framework:

|       Granularity       |    Backend     |   `io_type`   | `strategy` | Support |
| :---------------------: | :------------: | :-----------: | :--------: | :-----: |
|   Intra-layer, weight   | `Vivado/Vitis` |      \*       |  latency   |    ✅    |
|   Intra-layer, weight   |   `Quartus`    |      \*       |  latency   |    ❔    |
|   Intra-layer, weight   |       \*       |      \*       |  resource  |    ❌    |
| Intra-layer, activation | `Vivado/Vitis` | `io_parallel` |  latency   |    ✅    |
| Intra-layer, activation |      `*`       |  `io_stream`  |  latency   |    ❌    |
| Intra-layer, activation |   `Quartus`    |      \*       |  latency   |    ❌    |
| Intra-layer, activation |       \*       |      \*       |  resource  |    ❌    |
|       Inter-layer       |       \*       |      \*       |     \*     |    ✅    |
