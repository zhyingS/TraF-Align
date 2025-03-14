# TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception

The official implementation of paper:
> **TraF-Align: Trajectory-aware Feature Alignment for Asynchronous
Multi-agent Perception**           
>  [Zhiying Song](https://scholar.google.com/citations?user=joReSgYAAAAJ&hl=en), 
[Lei Yang](https://scholar.google.com/citations?hl=en&user=EUnI2nMAAAAJ), 
[Fuxi Wen](https://scholar.google.com/citations?user=gPsEbpgAAAAJ&hl=en)\*, Jun Li<br>
>  Tsinghua University <br>
>  Accepted to CVPR 2025

The code will be available in July, 2025.

## Abstract
Cooperative perception presents significant potential for enhancing the sensing capabilities of individual vehicles, however, inter-agent latency remains a critical challenge. Latencies cause misalignments in both spatial and semantic features, complicating the fusion of real-time observations from the ego vehicle with delayed data from others. To address these issues, we propose TraF-Align, a novel framework that learns the flow path of features by predicting the feature-level trajectory of objects from past observations up to the ego vehicle’s current time. By generating temporally ordered sampling points along these paths, TraF-Align directs attention from the current-time query to relevant historical features along each trajectory, supporting the reconstruction of current-time features and promoting semantic interaction across multiple frames. This approach corrects spatial misalignment and ensures semantic consistency across agents, effectively compensating for motion and achieving coherent feature fusion. Experiments on two real-world datasets, V2V4Real and DAIR-V2X-Seq, show that TraF-Align sets a new benchmark for asynchronous cooperative perception. 

## Citation
  ```bibtex
@InProceedings{Song_2025_Trafalign,
    author    = {Song, Zhiying and Yang, Lei and Wen, Fuxi and Li, Jun},
    title     = {TraF-Align: Trajectory-aware Feature Alignment for Asynchronous Multi-agent Perception},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2025},
    pages     = {}
}
```
