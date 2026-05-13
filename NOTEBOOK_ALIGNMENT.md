# Notebook Alignment Notes

Source notebook checked: `../Muad+NECO+energy.ipynb`.

The project is aligned to the notebook on these reproducibility-critical points:

- Data transforms: `Resize((256, 512), antialias=True)`, train `RandomHorizontalFlip`, `ToDtype(..., scale=True)`, ImageNet normalization, and MUAD `small` semantic splits.
- Training parameters: batch size 16, SGD learning rate 0.02, momentum 0.9, weight decay 5e-4, `StepLR(step_size=40, gamma=0.1)`, 100 epochs, 19 trained classes, ignore index 255, ENet weights from 21 labels sliced to 19.
- U-Net: module hierarchy and checkpoint keys match the notebook (`inc.conv.conv.*`, `down*.mpconv.*`, `up*.conv.conv.*`), with the notebook bilinear resize path in decoder up blocks.
- Default checkpoints/history: `checkpoints/unet.pth`, `checkpoints/unet_ens_*.pth`, and `results/training_history.json` are synchronized from the original notebook artifacts. 
- Training plot: `results/training_curves_unet.png` uses the notebook's colors, `figsize=(12, 5)`, subplot titles/labels, `tight_layout`, `dpi=200`, and `bbox_inches="tight"`.
- Paper figures: the script now regenerates the notebook-saved single-scene OOD qualitative figure, Deep Ensemble ECE, score distributions, H1 multi-scene qualitative figure, H2 condition-wise bars, and H3 ROC tail zoom with matching rcParams, colormaps, figure sizes, save names, DPI, and bbox settings. To avoid duplicate outputs, each figure is saved under one descriptive/notebook-style name only.

One intentional project-local difference remains: the MUAD cache root is `./datasets/muad` instead of the notebook's `./data`. This changes only where files are stored; the transform pipeline and split definitions are unchanged.
