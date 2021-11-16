#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace torchaudio {
namespace sentencepiece {

struct _SentencePieceProcessor : torch::CustomClassHolder {
  ::sentencepiece::SentencePieceProcessor processor_;

  std::string DecodePieces(const std::vector<std::string>& pieces) const {
    return processor_.DecodePieces(pieces);
  }

  void Load(const std::string& path) {
    processor_.Load(path);
  }
};

TORCH_LIBRARY(torchaudio, m) {
  m.class_<_SentencePieceProcessor>("SentencePieceProcessor")
      .def("DecodePieces", &_SentencePieceProcessor::DecodePieces)
    .def("Load", &_SentencePieceProcessor::Load);
  
}

} // namespace sentencepiece
} // namespace torchaudio
