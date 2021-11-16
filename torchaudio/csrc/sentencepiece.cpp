#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>
#include <torch/custom_class.h>
#include <torch/script.h>

namespace torchaudio {
namespace sentencepiece {
struct SentencePieceProcessor : torch::CustomClassHolder {
 private:
  sentencepiece::SentencePieceProcessor processor_;

  std::string DecodePieces(const std::vector<std::string>& pieces) const {
    return processor_.DecodePieces(pieces);
  }

  void Load(const std::string& path) {
    processor_.Load(path);
  }
}

TORCH_LIBRARY(torchaudio, m) {
  m.class_<SentencePieceProcessor>("SentencePieceProcessor")
      .def(torch::init<void>())
      .def("DecodePieces", &SentencePieceProcessor::DecodePieces)
      .def("Load", &SentencePieceProcessor::Load)
}

} // namespace sentencepiece
} // namespace torchaudio