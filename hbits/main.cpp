
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#include "gc_interface.h"
#include "tensor.h"
#include "tpc_test_core_api.h"
class MyTestKern;

static gcapi::HabanaKernelParams_t m_in_defs;
static gcapi::HabanaKernelInstantiation_t m_out_defs;
static constexpr int c_default_isa_buffer_size = 1024 * 1024;

#ifndef KERNEL_LIB
#define KERNEL_LIB KERNEL_LIB_UNDEFINED
#endif

// https://stackoverflow.com/a/5256500/9817693
#define _PPCAT_NX(A, B, C) A##B##C
#define _PPCAT(A, B, C) _PPCAT_NX(A, B, C)

// The kernel symble can be checked with nm ./lib<KERNEL_LIB>.a
#define kernel_symble_start _PPCAT(_binary___, KERNEL_LIB, _o_start)
#define kernel_symble_end _PPCAT(_binary___, KERNEL_LIB, _o_end)

extern unsigned char kernel_symble_start;
extern unsigned char kernel_symble_end;

static const std::string dataType[] = {"float32", "float16", "int32",   "int16",
                                       "int8",    "uint8",   "bfloat16"};

void PrintKernelInputParams(const gcapi::HabanaKernelParams_t *gc_input) {
  std::stringstream ss;
  ss << "Kernel Input Params:" << std::endl;
  ss << "\tinputTensorNr = " << gc_input->inputTensorNr << std::endl;
  for (unsigned i = 0; i < gc_input->inputTensorNr; i++) {
    ss << "\tinputTensors[" << i << "]."
       << dataType[gc_input->inputTensors[i].dataType] << "_"
       << gc_input->inputTensors[i].geometry.dims << "DTensor[] = {"
       << gc_input->inputTensors[i].geometry.sizes[0] << ", "
       << gc_input->inputTensors[i].geometry.sizes[1] << ", "
       << gc_input->inputTensors[i].geometry.sizes[2] << ", "
       << gc_input->inputTensors[i].geometry.sizes[3] << ", "
       << gc_input->inputTensors[i].geometry.sizes[4] << "}" << std::endl;
    if (gc_input->inputTensors[i].dataType != gcapi::DATA_F32) {
      ss << "\tinputTensors[" << i
         << "].scale = " << gc_input->inputTensors[i].quantizationParam.scale
         << std::endl;
      ss << "\tinputTensors[" << i << "].zeroPoint = "
         << (int)gc_input->inputTensors[i].quantizationParam.zeroPoint
         << std::endl;
    }
  }
  ss << "\toutputTensorNr = " << gc_input->outputTensorNr << std::endl;
  for (unsigned i = 0; i < gc_input->outputTensorNr; i++) {
    ss << "\toutputTensors[" << i << "]."
       << dataType[gc_input->outputTensors[i].dataType] << "_"
       << gc_input->outputTensors[i].geometry.dims << "DTensor[] = {"
       << gc_input->outputTensors[i].geometry.sizes[0] << ", "
       << gc_input->outputTensors[i].geometry.sizes[1] << ", "
       << gc_input->outputTensors[i].geometry.sizes[2] << ", "
       << gc_input->outputTensors[i].geometry.sizes[3] << ", "
       << gc_input->outputTensors[i].geometry.sizes[4] << "}" << std::endl;
    if (gc_input->inputTensors[i].dataType != gcapi::DATA_F32) {
      ss << "\toutputTensors[" << i
         << "].scale = " << gc_input->outputTensors[i].quantizationParam.scale
         << std::endl;
      ss << "\toutputTensors[" << i << "].zeroPoint = "
         << (int)gc_input->outputTensors[i].quantizationParam.zeroPoint
         << std::endl;
    }
  }
  ss << "\tdebugFlags = " << gc_input->debugFlags << std::endl << std::endl;

  std::cout << ss.str();
}

void PrintKernelOutputParams(
    const gcapi::HabanaKernelParams_t *gc_input,
    const gcapi::HabanaKernelInstantiation_t *gc_output) {
  std::stringstream ss;
  ss << "Glue code outputs:" << std::endl;
  ss << "\tindexSpaceGeometry.dims  = " << gc_output->indexSpaceGeometry.dims
     << std::endl;
  ss << "\tindexSpaceGeometry.sizes = "
     << gc_output->indexSpaceGeometry.sizes[0] << ", "
     << gc_output->indexSpaceGeometry.sizes[1] << ", "
     << gc_output->indexSpaceGeometry.sizes[2] << ", "
     << gc_output->indexSpaceGeometry.sizes[3] << ", "
     << gc_output->indexSpaceGeometry.sizes[4] << std::endl;
  for (unsigned i = 0; i < gc_input->inputTensorNr; i++) {
    ss << "\tinputTensorAccessPattern[" << i << "].allRequired = "
       << gc_output->inputTensorAccessPattern[i].allRequired << std::endl;

    for (unsigned j = 0; j < gc_output->indexSpaceGeometry.dims; j++) {
      ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].indexSpaceDim = "
         << gc_output->inputTensorAccessPattern[i].dim[j].dim << std::endl;
      ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].start_a = "
         << gc_output->inputTensorAccessPattern[i].dim[j].start_a << std::endl;
      ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].start_b = "
         << gc_output->inputTensorAccessPattern[i].dim[j].start_b << std::endl;
      ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].end_a = " << gc_output->inputTensorAccessPattern[i].dim[j].end_a
         << std::endl;
      ss << "\tinputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].end_b = " << gc_output->inputTensorAccessPattern[i].dim[j].end_b
         << std::endl;
    }
    ss << "\tinputPadValues[" << i
       << "].i32Value = " << gc_output->inputPadValues[i].i32Value << std::endl;
  }
  for (unsigned i = 0; i < gc_input->outputTensorNr; i++) {
    ss << "\toutputTensorAccessPattern[" << i << "].allRequired = "
       << gc_output->outputTensorAccessPattern[i].allRequired << std::endl;

    for (unsigned j = 0; j < gc_output->indexSpaceGeometry.dims; j++) {
      ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].indexSpaceDim = "
         << gc_output->outputTensorAccessPattern[i].dim[j].dim << std::endl;
      ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].start_a = "
         << gc_output->outputTensorAccessPattern[i].dim[j].start_a << std::endl;
      ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].start_b = "
         << gc_output->outputTensorAccessPattern[i].dim[j].start_b << std::endl;
      ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].end_a = " << gc_output->outputTensorAccessPattern[i].dim[j].end_a
         << std::endl;
      ss << "\toutputTensorAccessPattern[" << i << "].tensorDim[" << j
         << "].end_b = " << gc_output->outputTensorAccessPattern[i].dim[j].end_b
         << std::endl;
    }
  }
  ss << "\tauxiliaryTensorCount = " << gc_output->auxiliaryTensorCount
     << std::endl;
  ss << "\tkernel.kernelBinary = " << gc_output->kernel.kernelBinary
     << std::endl;
  ss << "\tkernel.binarySize = " << gc_output->kernel.binarySize << std::endl;
  ss << "\tkernel.paramsNr = " << gc_output->kernel.paramsNr << std::endl;
  for (unsigned i = 0; i < gc_output->kernel.paramsNr; i++) {
    ss << "\tkernel.scalarParams[" << i
       << "] = " << gc_output->kernel.scalarParams[i] << std::endl;
  }
  ss << "\tflags.Value = " << gc_output->flags.Value << std::endl;

  std::cout << ss.str();
}

int main(int argc, char const *argv[]) {
  printf("Hello hbits!\n");

  // clear in/out structures.
  memset(&m_in_defs, 0, sizeof(m_in_defs));
  memset(&m_out_defs, 0, sizeof(m_out_defs));

  // allocate memory for ISA
  const auto isa_buffer = std::unique_ptr<char>(
      reinterpret_cast<char *>(new char[c_default_isa_buffer_size]));
  m_out_defs.elfSize = c_default_isa_buffer_size;
  m_out_defs.kernelElf = isa_buffer.get();

  //////////////////////////////////////////////////////////

  const int width = 32;
  const int height = 32;

  unsigned fmInitializer[] = {width, height, 1, 1, 1};

  float_5DTensor input0(fmInitializer);
  input0.InitRand(0.f, 0.f);
  input0.FillWithData(0+1, 97+1);
  float_5DTensor input1(fmInitializer);
  input1.InitRand(0.f, 0.f);
  input1.FillWithData(0.f, 0.97f, 0.01f);
  input1.FillWithValue(2.f);

  float_5DTensor output(fmInitializer);
  output.InitRand(-999.f, -999.f);
  output.FillWithValue(-999);

  // generate input for query call
  m_in_defs.deviceId = gcapi::DEVICE_ID_GAUDI;

  // input tensors
  m_in_defs.inputTensorNr = 2;
  m_in_defs.inputTensors[0].dataType = gcapi::DATA_F32;
  m_in_defs.inputTensors[0].geometry.dims = 5;
  memcpy(m_in_defs.inputTensors[0].geometry.sizes, fmInitializer,
         sizeof(fmInitializer));
  m_in_defs.inputTensors[1].dataType = gcapi::DATA_F32;
  m_in_defs.inputTensors[1].geometry.dims = 5;
  memcpy(m_in_defs.inputTensors[1].geometry.sizes, fmInitializer,
         sizeof(fmInitializer));

  // output tensors
  m_in_defs.outputTensorNr = 1;
  m_in_defs.outputTensors[0].dataType = gcapi::DATA_F32;
  m_in_defs.outputTensors[0].geometry.dims = 5;
  memcpy(m_in_defs.outputTensors[0].geometry.sizes, fmInitializer,
         sizeof(fmInitializer));

  strncpy(m_in_defs.nodeName, "my_test_kern", gcapi::MAX_NODE_NAME);

  /*************************************************************************************
   *    Stage II -  Define index space geometry. In this example the index space
   *matches the dimensions of the output tensor, up to dim 0.
   **************************************************************************************/
  int elementsInVec = 64;
  unsigned int outputSizes[gcapi::MAX_TENSOR_DIM] = {0};
  memcpy(outputSizes, m_in_defs.inputTensors[0].geometry.sizes,
         sizeof(outputSizes));

  // round up to elementsInVec and divide by elementsInVec.
  //  i.e. 3x3 in
  //  https://docs.habana.ai/en/latest/TPC/TPC_User_Guide/TPC_Programming_Model.html#figure-2-4
  unsigned depthIndex = (outputSizes[0] + (elementsInVec - 1)) / elementsInVec;
  m_out_defs.indexSpaceGeometry.dims = 5;
  m_out_defs.indexSpaceGeometry.sizes[0] = depthIndex;
  m_out_defs.indexSpaceGeometry.sizes[1] = outputSizes[1];
  m_out_defs.indexSpaceGeometry.sizes[2] = outputSizes[2];
  m_out_defs.indexSpaceGeometry.sizes[3] = outputSizes[3];
  m_out_defs.indexSpaceGeometry.sizes[4] = outputSizes[4];

  /*************************************************************************************
   *    Stage III -  Define index space mapping
   **************************************************************************************/

  // Index space mapping is calculated using f(i) = Ai + B
  // 'i' is the index space member and A/B constants to be defined.
  m_out_defs.inputTensorAccessPattern[0].dim[0].dim = 0;
  m_out_defs.inputTensorAccessPattern[0].dim[0].start_a = elementsInVec;
  m_out_defs.inputTensorAccessPattern[0].dim[0].end_a = elementsInVec;
  m_out_defs.inputTensorAccessPattern[0].dim[0].start_b = 0;
  m_out_defs.inputTensorAccessPattern[0].dim[0].end_b = elementsInVec - 1;

  // f_start f(i) = 1*i + 0;
  // f_end   f(i) = 1*i + 0;
  // Resource 0 (IFM) dim 1-4
  for (int dims = 3; dims < 5; dims++) {
    m_out_defs.inputTensorAccessPattern[0].dim[dims].dim = dims;
    m_out_defs.inputTensorAccessPattern[0].dim[dims].start_a = 1;
    m_out_defs.inputTensorAccessPattern[0].dim[dims].end_a = 1;
    m_out_defs.inputTensorAccessPattern[0].dim[dims].start_b = 0;
    m_out_defs.inputTensorAccessPattern[0].dim[dims].end_b = 1 - 1;
  }

  // dst
  for (int i = 0; i < 5; ++i)
    m_out_defs.outputTensorAccessPattern[0].dim[i] =
        m_out_defs.inputTensorAccessPattern[0].dim[i];

  /*************************************************************************************
   *    Stage V -  Load ISA into the descriptor.
   **************************************************************************************/
  unsigned isa_size = (&kernel_symble_end - &kernel_symble_start);
  assert(m_out_defs.elfSize >= isa_size);
  m_out_defs.elfSize = isa_size;
  // copy binary out
  memcpy(m_out_defs.kernelElf, &kernel_symble_start, isa_size);

  // generate and load tensor descriptors
  std::vector<tpc_tests::TensorDesc> tds;
  tds.push_back(input0.GetTensorDescriptor());
  tds.push_back(input1.GetTensorDescriptor());
  tds.push_back(output.GetTensorDescriptor());
  // execute a simulation of the kernel using TPC simulator,

  // debug prints of glue code input and output.
  PrintKernelInputParams(&m_in_defs);
  PrintKernelOutputParams(&m_in_defs, &m_out_defs);

  const auto retVal = tpc_tests::RunSimulation(m_in_defs, m_out_defs, tds);
  (void)(retVal);

  // output.Print(0);
  std::cout << std::endl;
  for (int d2 = 0; d2 < height; d2++) {
    for (int d1 = 0; d1 < width; d1++) {
      int coords[] = {d1, d2, 0, 0, 0};

      std::cout << std::fixed << std::setw(5) << std::setprecision(2)
                << std::setfill(' ') << (float)output.ElementAt(coords) << ",";
    }
    std::cout << std::endl;
  }
  return retVal;
}
