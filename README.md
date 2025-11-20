# Human3R-infer (TPU Version)

This project provides a streamlined, TPU-compatible inference implementation of the Human3R model. It has been refactored to remove CUDA dependencies and optimize for XLA execution.

## Key Features

*   **TPU/XLA Support**: Native support for TPU inference using `torch_xla`.
*   **Pure Python RoPE**: Replaced custom CUDA kernels for Rotary Positional Embeddings with a pure Python/PyTorch implementation.
*   **Simplified Architecture**: Removed rendering dependencies and complex file structures.
*   **Modular Design**: Split into clear, functional modules for easier maintenance.

## Project Structure

*   **`modules.py`**: Contains foundational building blocks (Attention, Block, MLP, RoPE, PatchEmbed).
*   **`heads.py`**: Defines prediction heads (DPT, SMPL Decoder, Pose Decoder).
*   **`model.py`**: Implements the core backbone (`Dinov2`) and top-level model (`ARCroco3DStereo`).
*   **`utils.py`**: Provides geometric calculations, camera transformations, and image preprocessing (rendering removed).
*   **`run_tpu.py`**: The main entry point for running inference on TPU.

## Requirements

Ensure your environment is set up with the following:

*   Python 3.8+
*   PyTorch
*   `torch_xla` (for TPU support)
*   `einops`
*   `roma`
*   `timm`
*   `Pillow`
*   `numpy`

## Usage

1.  **Prepare Weights**: Upload your Human3R `.pth` weights to your TPU VM or accessible path.
2.  **Run Inference**:

```bash
python3 run_tpu.py \
    --model_path /path/to/human3r.pth \
    --input_image /path/to/image.jpg \
    --output_path result.pt
```

## Output

The script saves the inference results as a `.pt` file containing:
*   `pts3d`: 3D point cloud coordinates.
*   `conf`: Confidence scores.
*   (And other model-specific outputs)

## Notes

*   **Rendering**: This version does not generate mesh visualizations directly to avoid `pytorch3d` or other rendering dependencies on TPU. It outputs raw parameters and vertices.
*   **Utils**: Some utility functions (`nms`, `apply_threshold`, `unpad_uv`) in `utils.py` are currently placeholders and may need implementation depending on your specific post-processing needs.