//
// Created by Leo Li on 2019-11-14.
//

#include "libethash-cuda/CUDAMiner.h"
#include "libethash-cuda/ethash_cuda_miner_kernel.h"
#include "libethcore/Miner.h"
#include <boost/asio/io_service.hpp>

using namespace dev::eth;


bool g_exitOnError = false;  // Whether or not ethminer should exit on mining threads errors

boost::asio::io_service g_io_service;  //


int main() {
    const int num_streams = 8;
    cudaStream_t streams[num_streams];
    volatile Search_results *buffer = nullptr;
    CUDA_SAFE_CALL(cudaMallocHost(&buffer, sizeof(Search_results)));
    buffer->count = 0;
    for (unsigned i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    CUSettings setting;
    std::map<std::string, DeviceDescriptor> m_DevicesCollection = {};
    CUDAMiner::enumDevices(m_DevicesCollection);
    CUDAMiner miner(0, setting, m_DevicesCollection.begin()->second);
    WorkPackage wp;
    auto _ec = ethash::get_global_epoch_context(0);
    EpochContext c;
    c.epochNumber = 0;
    c.lightNumItems = _ec.light_cache_num_items;
    c.lightSize = ethash::get_light_cache_size(_ec.light_cache_num_items);
    c.dagNumItems = _ec.full_dataset_num_items;
    c.dagSize = ethash::get_full_dataset_size(_ec.full_dataset_num_items);
    c.lightCache = _ec.light_cache;
    miner.setEpoch(c);
    miner.workLoop();
}