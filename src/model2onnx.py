import torch

from model import SegmentationModel

def load_and_save_onnx(ckpt, output_path="model.onnx", input_shape=(1, 3, 512, 512)):
    """Load a trained model and save it as ONNX format."""
    model = SegmentationModel.load_from_checkpoint(ckpt)
    model.eval()
    dummy_input = torch.randn(input_shape).to(model.device)  
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        input_names=["input"], 
        output_names=["output"], 
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX format")
    parser.add_argument("checkpoint", type=str, help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output ONNX file path")
    parser.add_argument("--input_shape", type=int, nargs=4, default=(1, 3, 512, 512), help="Input shape for the model")

    args = parser.parse_args()
    
    load_and_save_onnx(args.checkpoint, args.output, tuple(args.input_shape))