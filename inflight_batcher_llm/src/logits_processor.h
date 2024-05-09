//
// Created by pusl on 2024/5/2.
//

#ifndef TRITONTENSORRTLLMBACKEND_LOGITS_PROCESSOR_H
#define TRITONTENSORRTLLMBACKEND_LOGITS_PROCESSOR_H

#include <cstdint>
#include <vector>
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/batch_manager/inferenceRequest.h"

#define MaxStrategyNums 10

// 命名规范: https://google.github.io/styleguide/cppguide.html#General_Naming_Rules


namespace triton::backend::soul_strategy {
    using namespace std;
    namespace tr = tensorrt_llm::runtime;
    using InferenceRequest = tensorrt_llm::batch_manager::InferenceRequest;

    using TokenIdType = tensorrt_llm::runtime::TokenIdType;
    using VecTokens = std::vector<TokenIdType>;
    using TStream = tensorrt_llm::runtime::BufferManager::CudaStreamPtr;
    using RequestIdType = std::uint64_t;
    using TensorPtr = tensorrt_llm::runtime::ITensor::SharedPtr;
    using BeamTokens = std::vector<VecTokens>;

    struct TokenSuppressData {
        // todo: change to gpu tensor
        int *token_ids_;
        float *scales_;

        TokenSuppressData(int *token_ids, float *scales, int nums) {
            token_ids_ = new int[nums];
            scales_ = new float[nums];
            copy(token_ids, token_ids + nums, token_ids_);
            copy(scales, scales + nums, scales_);
        };

        ~TokenSuppressData() {
            delete token_ids_;
            delete scales_;
        };

    };

    class LogitsProcessorStrategy {
    public:
        explicit LogitsProcessorStrategy(int *strategies) {
            strategies_ = new int[MaxStrategyNums];
            copy(strategies, strategies + MaxStrategyNums, strategies_);

            token_suppress_data_ = nullptr;
        };

        ~LogitsProcessorStrategy() {
            delete strategies_;
            delete token_suppress_data_;
        }

        void setLogitsProcessor(const std::shared_ptr<InferenceRequest>& ir) {
            // todo: register other strategy function
            ir->setLogitsPostProcessor(
                    [this](RequestIdType req_id_type, TensorPtr &t_ptr, BeamTokens const &beam_tokens,
                           const TStream& t_stream) {
                        return token_suppress_strategy(req_id_type, t_ptr, beam_tokens, t_stream);
                    });
        };

        void setTokenSuppressData(int *token_ids, float *scales, int nums) {
            token_suppress_data_ = new TokenSuppressData(token_ids, scales, nums);
        }

    private:
        void token_suppress_strategy(
                RequestIdType req_id_type, TensorPtr &t_ptr, BeamTokens const &beam_tokens, const TStream& t_stream
        ) {
            std::printf(">>> Request id type %lu\n", req_id_type);
            std::printf(">>> Beam tokens at zero is %d \n", beam_tokens.at(0).at(0));
            std::printf(">>> beam token size %zu \n", beam_tokens.at(0).size());
        };

    private:
        // actually a list
        int *strategies_;
        TokenSuppressData *token_suppress_data_;
    };
}

#endif //TRITONTENSORRTLLMBACKEND_LOGITS_PROCESSOR_H
