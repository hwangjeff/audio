#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace torchaudio {
namespace sentencepiece {

struct SentencePieceProcessor : torch::CustomClassHolder {
  ::sentencepiece::SentencePieceProcessor processor_;

  SentencePieceProcessor() {}

  std::string DecodePieces(const std::vector<std::string>& pieces) const {
    return processor_.DecodePieces(pieces);
  }

  void Load(const std::string& path) {
    processor_.Load(path);
  }
};

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.class_<SentencePieceProcessor>("SentencePieceProcessor")
      // .def(torch::init())
      .def("DecodePieces", &SentencePieceProcessor::DecodePieces)
      .def("Load", &SentencePieceProcessor::Load)
      .def_pickle(
          // __getstate__
          [](const c10::intrusive_ptr<SentencePieceProcessor>& self)
              -> torch::Tensor {
            auto serialized_model = self->processor_.serialized_model_proto();
            auto* data =
                static_cast<void*>(const_cast<char*>(serialized_model.data()));
            auto numel = static_cast<int64_t>(serialized_model.size());
            return torch::from_blob(data, {numel}, {torch::kUInt8}).clone();
          },
          // __setstate__
          [](torch::Tensor state)
              -> c10::intrusive_ptr<SentencePieceProcessor> {
            auto* data = static_cast<char*>(state.data_ptr());
            auto numel = state.size(0);
            auto processor = c10::make_intrusive<SentencePieceProcessor>();
            processor->processor_.LoadFromSerializedProto(
                std::string(data, numel));
            return processor;
          });
}

c10::intrusive_ptr<SentencePieceProcessor> init_sentencepiece_processor() {
  return c10::make_intrusive<SentencePieceProcessor>();
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def("init_sentencepiece_processor", init_sentencepiece_processor);
}

} // namespace sentencepiece
} // namespace torchaudio
