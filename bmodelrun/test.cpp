
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "tpuv7_rt.h"
#include "tpuv7_modelrt.h"

static tpuRtStream_t m_stream;

uint64_t shapeCount(tpuRtShape_t shape) {
    uint64_t count = 1;
    for (int i = 0; i < shape.num_dims; i++) {
	count *= shape.dims[i];
    }
    return count;
  }

void freeTensor(std::vector<tpuRtTensor_t> &tensor) {
    for (auto iter : tensor) {
      tpuRtFree(&iter.data, 0);
    }
  }

int dataTypeSize(tpuRtDataType_t dtype) {
    switch (dtype) {
      case TPU_FLOAT32:
      case TPU_INT32:
      case TPU_UINT32:
        return 4;
      case TPU_FLOAT16:
      case TPU_BFLOAT16:
      case TPU_INT16:
        return 2;
      case TPU_INT8:
      case TPU_UINT8:
        return 1;
      case TPU_INT4:
      case TPU_UINT4:
        return 1;  // need modify ?  to do
      default:
        return 4;
    }
  }

void mallocTensor(std::vector<tpuRtTensor_t> &tensor,
                                   const tpuRtIOInfo_t &info,
                                   const tpuRtShape_t *shape) {
    for (int i = 0; i < info.num; ++i) {
      tensor[i].dtype = info.dtypes[i];
      tensor[i].shape.num_dims = shape[i].num_dims;
      memcpy(tensor[i].shape.dims, shape[i].dims,
             sizeof(int) * shape[i].num_dims);
      tpuRtMalloc(&tensor[i].data, shapeCount(shape[i]) * dataTypeSize(info.dtypes[i]), 0);
    }
  }

void run_net(tpuRtNet_t *net, const char *netname){
    auto m_net = *net;
    auto m_info = tpuRtGetNetInfo(m_net, netname);
    auto net_info = m_info;
    int num_input = m_info.input.num;
    int num_output = m_info.output.num;
    int stage = 0;
    std::vector<void *> input_datas;
    std::vector<void *> output_datas;
    std::vector<tpuRtTensor_t> input_tensors(net_info.input.num);
    std::vector<tpuRtTensor_t> output_tensors(net_info.output.num);
    mallocTensor(input_tensors, net_info.input, net_info.stages[0].input_shapes);
    mallocTensor(output_tensors, net_info.output, net_info.stages[0].output_shapes);
    for (int i = 0; i < num_input; i++) {
        auto &shape = m_info.stages[stage].input_shapes[i];
        auto &dtype = m_info.input.dtypes[i];
        auto size = shapeCount(shape) * dataTypeSize(dtype);
        void *data;
        tpuRtMallocHost(&data, size);
        input_datas.push_back(data);
        tpuRtMemcpyS2D(input_tensors[i].data, data, size);
    }
    // prepare input data
    tpuRtLaunchNet(m_net, input_tensors.data(), output_tensors.data(), netname, m_stream);
    for (int i = 0; i < num_output; i++) {
        auto &shape = m_info.stages[stage].output_shapes[i];
        auto &dtype = m_info.output.dtypes[i];
        auto size = shapeCount(shape) * dataTypeSize(dtype);
        void *data;
        tpuRtMallocHost(&data, size);
        output_datas.push_back(data);
        tpuRtMemcpyD2S(data, output_tensors[i].data, size);
    }
    freeTensor(input_tensors);
    freeTensor(output_tensors);
    for(auto &data : input_datas){
        tpuRtFreeHost(data);
    }
    for(auto &data : output_datas){
        tpuRtFreeHost(data);
    }
}

void run_bmodel(tpuRtNet_t *net){
    // prepare input data 
    char **net_names = NULL;
    int net_num = tpuRtGetNetNames(*net, &net_names);
    for (int i = 0; i < net_num; i++){
        std::cout << "run net: " << net_names[i] << std::endl;
        run_net(net, net_names[i]);
    }
    tpuRtFreeNetNames(net_names);
}

int main()
{
    const char* bmodel_path2 = "/home/zwy/SD3_Perf_Zuliang/PipelineModels/clipg/clip_g.bmodel";
    const char* bmodel_path1 = "/home/zwy/SD3_Perf_Zuliang/PipelineModels/clipl/clip_l.bmodel";
    const char* bmodel_path3 = "/home/zwy/SD3_Perf_Zuliang/PipelineModels/t5/t5.bmodel";
    const char* bmodel_path4 = "/home/zwy/SD3_Perf_Zuliang/PipelineModels/mmdit/mmdit.bmodel";

    tpuRtInit();
    tpuRtSetDevice(0);
    
    tpuRtNetContext_t m_context;
    tpuRtCreateNetContext(&m_context);
    tpuRtNet_t m_net1;
    tpuRtNet_t m_net2;
    tpuRtNet_t m_net3;
    tpuRtNet_t m_net4;
    tpuRtLoadNet(bmodel_path1, m_context, &m_net1);
    tpuRtLoadNet(bmodel_path2, m_context, &m_net2);
    tpuRtLoadNet(bmodel_path3, m_context, &m_net3);
    tpuRtLoadNet(bmodel_path4, m_context, &m_net4);
    tpuRtStreamCreate(&m_stream);

    run_bmodel(&m_net1);
    run_bmodel(&m_net2);
    run_bmodel(&m_net3);
    run_bmodel(&m_net4);

    tpuRtUnloadNet(m_net1);
    tpuRtUnloadNet(m_net2);
    tpuRtUnloadNet(m_net3);
    tpuRtUnloadNet(m_net4);

    tpuRtDestroyNetContext(m_context);
    tpuRtStreamDestroy(m_stream);
    return 0;
}

int main2()
{
  std::vector<std::string> bmodel_paths;
  bmodel_paths.push_back("/home/zwy/SD3_Perf_Zuliang/PipelineModels/clipg/clip_g.bmodel");
  bmodel_paths.push_back("/home/zwy/SD3_Perf_Zuliang/PipelineModels/clipl/clip_l.bmodel");
  bmodel_paths.push_back("/home/zwy/SD3_Perf_Zuliang/PipelineModels/t5/t5.bmodel");
  bmodel_paths.push_back("/home/zwy/SD3_Perf_Zuliang/PipelineModels/mmdit/mmdit.bmodel");
  bmodel_paths.push_back("/home/zwy/SD3_Perf_Zuliang/PipelineModels/vae/vae_decoder.bmodel");
  tpuRtInit();
  tpuRtSetDevice(0);
  
  tpuRtNetContext_t m_context;
  tpuRtCreateNetContext(&m_context);

  std::vector<tpuRtNet_t> m_nets;
  
  for (auto bmodel_path : bmodel_paths){
    tpuRtNet_t m_net;
    tpuRtLoadNet(bmodel_path.c_str(), m_context, &m_net);
    m_nets.push_back(m_net);
  }

  tpuRtStream_t m_stream;
  tpuRtStreamCreate(&m_stream);

  for (auto m_net : m_nets){
    run_bmodel(&m_net);
  }

  for (auto m_net : m_nets){
    tpuRtUnloadNet(m_net);
  }

  tpuRtDestroyNetContext(m_context);
  tpuRtStreamDestroy(m_stream);
  return 0;
}