import torch.onnx
from training.model import SentenceClassificationModule
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch as t

model = SentenceClassificationModule(512, 64)


def restore_checkpoint(epoch_n):
    ckp = t.load('../artefacts/checkpoint_{:03d}.ckp'.format(epoch_n), None)

    model.load_state_dict(ckp['state_dict'])


restore_checkpoint(100)


model.eval()

batch_size = 1

sample_input = torch.randn(batch_size, 1, 512, requires_grad=True)

output = model(sample_input)

model_name = "flexudy_grammar_checker.onnx"

# Export the model
torch.onnx.export(model,
                  sample_input,
                  model_name,
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

quantized_model_name = 'flexudy_sentence_scorer.quantized.onnx'

quantized_model = quantize_dynamic(model_name, quantized_model_name, weight_type=QuantType.QUInt8)