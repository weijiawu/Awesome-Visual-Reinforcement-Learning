# Awesome-Visual-Reinforcement-Learning
📖 This is a repository for organizing papers, codes and other resources related to Visual Reinforcement Learning.


<p align="center">
  <img src="assets/RL.png" alt="TAX" style="display: block; margin: 0 auto;" width="300px" />
</p>

#### :thinking: What is Visual Reinforcement Learning?

**Visual Reinforcement Learning (Visual RL)** enables agents to learn decision-making policies directly from visual observations (e.g., images or videos), rather than structured state inputs.
It lies at the intersection of reinforcement learning and computer vision, with applications in robotics, embodied AI, games, and interactive environments.

#### 📌 Project Description
Awesome-Visual-Reinforcement-Learning is a curated list of papers, libraries, and resources on learning control policies from visual input.
It aims to help researchers and practitioners navigate the fast-evolving Visual RL landscape — from perception and representation learning to policy learning and real-world applications.


## 📚 Table of Contents <!-- omit in toc -->
Libraries and tools

- [Benchmarks environments and datasets with Visual RL](#benchmarks-environments-and-datasets-with-visual-rl)
- [Visual Perception with RL](#visual-perception-with-rl)
- [Multi-Modal Large Language Models with RL](#multi-modal-large-language-models-with-rl)
- [Visual Agents with RL](#visual-agents-with-rl)
- [Visual Generation with RL](#visual-generation-with-rl)
- [RL for Unified Model](#rl-for-unified-model)
- [Visual World Models with RL](#visual-world-models-with-rl)
- [RL for Embodied AI/Robotics](#rl-for-embodied-ai/robotics)
- [RL for Medical Reasoning](#rl-for-medical-reasoning)
- [Audio Question Answering with RL](#audio-question-answering-with-rl)
- [Others](#others)


### Benchmarks environments and datasets with Visual RL

#### MLLM

+ [MM-Eureka: Exploring the Frontiers of Multimodal Reasoning with Rule-based Reinforcement Learning](https://arxiv.org/abs/2503.07365) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.07365)
  [![Star](https://img.shields.io/github/stars/ModalMinds/MM-EUREKA.svg?style=social&label=Star)](https://github.com/ModalMinds/MM-EUREKA)


+ [Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1](https://arxiv.org/pdf/2503.24376) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.24376)
  [![Star](https://img.shields.io/github/stars/TencentARC/SEED-Bench-R1.svg?style=social&label=Star)](https://github.com/TencentARC/SEED-Bench-R1)

### Multi-Agent

+ [FightLadder: A Benchmark for Competitive Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2406.02081) (2024, ICML)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2406.02081)
  [![Star](https://img.shields.io/github/stars/wenzhe-li/FightLadder.svg?style=social&label=Star)](https://github.com/wenzhe-li/FightLadder)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://sites.google.com/view/fightladder/home)


### Visual Perception with RL
**Definition**: Focus on learning effective visual representations—including segmentation, depth estimation, and object recognition—from pixel inputs to guide RL agent decision-making.

+ [Optimization-Free Patch Attack on Stereo Depth Estimation](https://arxiv.org/abs/2506.17632) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.17632)

+ [VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2504.06958) (Apr. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.06958)

+ [Omni-R1: Reinforcement Learning for Omnimodal Reasoning via Two-System Collaboration](https://arxiv.org/pdf/2505.20256) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20256)
  [![Star](https://img.shields.io/github/stars/aim-uofa/Omni-R1.svg?style=social&label=Star)](https://github.com/aim-uofa/Omni-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://aim-uofa.github.io/OmniR1/)


+ [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.22019) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.22019)
  [![Star](https://img.shields.io/github/stars/Alibaba-NLP/VRAG.svg?style=social&label=Star)](https://github.com/Alibaba-NLP/VRAG)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://modomodo-rl.github.io/)


+ [VisualQuality-R1: Reasoning-Induced Image Quality Assessment via Reinforcement Learning to Rank](https://arxiv.org/pdf/2505.14460) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.14460)
  [![Star](https://img.shields.io/github/stars/TianheWu/VisualQuality-R1.svg?style=social&label=Star)](https://github.com/TianheWu/VisualQuality-R1)


+ [Integrating Saliency Ranking and Reinforcement Learning for Enhanced Object Detection](https://arxiv.org/abs/2408.06803) (Aug. 2024, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2408.06803)
  [![Star](https://img.shields.io/github/stars/mbar0075/SaRLVision.svg?style=social&label=Star)](https://github.com/mbar0075/SaRLVision)


+ [Perception-R1: Pioneering Perception Policy with Reinforcement Learning](https://arxiv.org/pdf/2504.07954) (Apr. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.07954)
  [![Star](https://img.shields.io/github/stars/linkangheng/PR1.svg?style=social&label=Star)](https://github.com/linkangheng/PR1)


+ [VisionReasoner: Unified Visual Perception and Reasoning via Reinforcement Learning](https://arxiv.org/abs/2505.12081) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.12081)
  [![Star](https://img.shields.io/github/stars/dvlab-research/VisionReasoner.svg?style=social&label=Star)](https://github.com/dvlab-research/VisionReasoner)


+ [Grounded Reinforcement Learning for Visual Reasoning](https://arxiv.org/abs/2505.23678) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23678)
  [![Star](https://img.shields.io/github/stars/Gabesarch/grounded-rl.svg?style=social&label=Star)](https://github.com/Gabesarch/grounded-rl)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://visually-grounded-rl.github.io/)


+ [UniVG-R1: Reasoning Guided Universal Visual Grounding with Reinforcement Learning](https://arxiv.org/abs/2505.14231) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.14231)
  [![Star](https://img.shields.io/github/stars/AMAP-ML/UniVG-R1.svg?style=social&label=Star)](https://github.com/AMAP-ML/UniVG-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://amap-ml.github.io/UniVG-R1-page/)


+ [Visual-RFT: Visual Reinforcement Fine-Tuning](https://arxiv.org/abs/2503.01785) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.01785)
  [![Star](https://img.shields.io/github/stars/Liuziyu77/Visual-RFT.svg?style=social&label=Star)](https://github.com/Liuziyu77/Visual-RFT)

### Multi-Modal Large Language Models with RL

#### Spatial Reasoning \& Verification

+ [SATORI-R1: Incentivizing Multimodal Reasoning with Spatial Grounding and Verifiable Rewards](https://arxiv.org/abs/2505.19094) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19094)
  [![Star](https://img.shields.io/github/stars/justairr/SATORI-R1.svg?style=social&label=Star)](https://github.com/justairr/SATORI-R1)

+ [Skywork R1V2: Multimodal Hybrid Reinforcement Learning for Reasoning](https://arxiv.org/pdf/2504.16656) (Apr. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.16656)
  [![Star](https://img.shields.io/github/stars/SkyworkAI/Skywork-R1V.svg?style=social&label=Star)](https://github.com/SkyworkAI/Skywork-R1V)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/Skywork/Skywork-R1V2-38B)


+ [Pixel Reasoner: Incentivizing Pixel-Space Reasoning with Curiosity-Driven Reinforcement Learning](https://arxiv.org/abs/2505.15966) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15966)
  [![Star](https://img.shields.io/github/stars/TIGER-AI-Lab/Pixel-Reasoner.svg?style=social&label=Star)](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tiger-ai-lab.github.io/Pixel-Reasoner)


+ [ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs](https://arxiv.org/pdf/2506.10128) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.10128)
  [![Star](https://img.shields.io/github/stars/si0wang/ViCrit.svg?style=social&label=Star)](https://github.com/si0wang/ViCrit)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/datasets/zyang39/ViCrit-Train)


+ [Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning](https://arxiv.org/pdf/2503.18013) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.18013)
  [![Star](https://img.shields.io/github/stars/jefferyZhan/Griffon.svg?style=social&label=Star)](https://github.com/jefferyZhan/Griffon)


+ [Visionary-R1: Mitigating Shortcuts in Visual Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.14677) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.14677)
  [![Star](https://img.shields.io/github/stars/lll6gg/UI-R1.svg?style=social&label=Star)](https://github.com/lll6gg/UI-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://www.maifoundations.com/blog/visionary-r1/)


+ [RQwen Look Again: Guiding Vision-Language Reasoning Models to Re-attention Visual Information](https://www.arxiv.org/abs/2505.23558) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.arxiv.org/abs/2505.23558)
  [![Star](https://img.shields.io/github/stars/Liar406/Look_Again.svg?style=social&label=Star)](https://github.com/Liar406/Look_Again)


+ [EasyARC: Evaluating Vision Language Models on True Visual Reasoning](https://arxiv.org/abs/2506.11595) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.11595)

+ [Aha Moment Revisited: Are VLMs Truly Capable of Self Verification in Inference-time Scaling?](https://arxiv.org/abs/2506.17417) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.17417)

+ [STAR-R1: Spatial TrAnsformation Reasoning by Reinforcing Multimodal LLMs](https://arxiv.org/pdf/2505.15804) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.15804)
  [![Star](https://img.shields.io/github/stars/zongzhao23/STAR-R1.svg?style=social&label=Star)](https://github.com/zongzhao23/STAR-R1)

+ [Perceptual Decoupling for Scalable Multi-modal Reasoning via Reward-Optimized Captioning](https://arxiv.org/abs/2506.04559) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.04559)

+ [DIP-R1: Deep Inspection and Perception with RL Looking Through and Understanding Complex Scenes](https://arxiv.org/abs/2505.23179) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23179)

+ [VL-GenRM: Enhancing Vision-Language Verification via Vision Experts and Iterative Training](https://arxiv.org/abs/2506.13888) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.13888)

+ [SVQA-R1: Reinforcing Spatial Reasoning in MLLMs via View-Consistent Reward Optimization](https://arxiv.org/abs/2506.01371) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01371)

+ [Advancing Multimodal Reasoning Capabilities of Multimodal Large Language Models via Visual Perception Reward](https://arxiv.org/abs/2506.07218) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.07218)

+ [Reinforcing Spatial Reasoning in Vision-Language Models with Interwoven Thinking and Visual Drawing](https://arxiv.org/abs/2506.09965) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.09965)

+ [Ground-R1: Incentivizing Grounded Visual Reasoning via Reinforcement Learning](https://arxiv.org/pdf/2505.20272) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20272)



#### Video \& Temporal Understanding

+ [TimeMaster: Training Time-Series Multimodal LLMs to Reason via Reinforcement Learning](https://arxiv.org/abs/2506.13705) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.13705)


+ [VAU-R1: Advancing Video Anomaly Understanding via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.23504) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.23504)
  [![Star](https://img.shields.io/github/stars/GVCLab/VAU-R1.svg?style=social&label=Star)](https://github.com/GVCLab/VAU-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://github.com/GVCLab/VAU-R1?tab=readme-ov-file)

+ [EgoVLM: Policy Optimization for Egocentric Video Understanding](https://arxiv.org/abs/2506.03097) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.03097)
  [![Star](https://img.shields.io/github/stars/adityavavre/VidEgoVLM.svg?style=social&label=Star)](https://github.com/adityavavre/VidEgoVLM)


+ [VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning](https://arxiv.org/abs/2506.18564) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.18564)

+ [Reinforcing Video Reasoning with Focused Thinking](https://arxiv.org/abs/2505.24718) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24718)
  [![Star](https://img.shields.io/github/stars/longmalongma/TW-GRPO.svg?style=social&label=Star)](https://github.com/longmalongma/TW-GRPO)

+ [Reinforcement Learning Tuning for VideoLLMs: Reward Design and Data Efficiency](https://arxiv.org/abs/2506.01908) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01908)
  [![Star](https://img.shields.io/github/stars/appletea233/Temporal-R1.svg?style=social&label=Star)](https://github.com/appletea233/Temporal-R1)


+ [DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO](https://arxiv.org/pdf/2506.07464) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.07464)
  [![Star](https://img.shields.io/github/stars/mlvlab/DeepVideoR1.svg?style=social&label=Star)](https://github.com/mlvlab/DeepVideoR1)


#### Goal-Driven \& Personalized Learning

+ [Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles](https://arxiv.org/abs/2505.23590) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23590)
  [![Star](https://img.shields.io/github/stars/zifuwanggg/Jigsaw-R1.svg?style=social&label=Star)](https://github.com/zifuwanggg/Jigsaw-R1)

+ [Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment](https://arxiv.org/pdf/2506.05384) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.05384)
  [![Star](https://img.shields.io/github/stars/vivoCameraResearch/Q-Ponder.svg?style=social&label=Star)](https://github.com/vivoCameraResearch/Q-Ponder)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vivocameraresearch.github.io/qponder/)

+ [MMSearch-R1: Incentivizing LMMs to Search](https://arxiv.org/abs/2506.20670) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.20670)
  [![Star](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1.svg?style=social&label=Star)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)


+ [VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model](https://arxiv.org/pdf/2504.07615) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.07615)
  [![Star](https://img.shields.io/github/stars/om-ai-lab/VLM-R1.svg?style=social&label=Star)](https://github.com/om-ai-lab/VLM-R1)


+ [WeThink: Toward General-purpose Vision-Language Reasoning via Reinforcement Learning](https://arxiv.org/abs/2506.07905) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.07905)


+ [Self-Rewarding Large Vision-Language Models for Optimizing Prompts in Text-to-Image Generation](https://arxiv.org/pdf/2505.16763) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.16763)


+ [Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning](https://arxiv.org/abs/2506.18234) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.18234)


+ [Play to Generalize: Learning to Reason Through Game Play](https://arxiv.org/abs/2506.08011) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.08011)
  [![Star](https://img.shields.io/github/stars/yunfeixie233/ViGaL.svg?style=social&label=Star)](https://github.com/yunfeixie233/ViGaL)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yunfeixie233.github.io/ViGaL/)

+ [GoalLadder: Incremental Goal Discovery with Vision-Language Models](https://arxiv.org/abs/2506.16396) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.16396)

+ [DeepEyes: Incentivizing “Thinking with Images” via Reinforcement Learning](https://arxiv.org/abs/2505.14362) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.14362)
  [![Star](https://img.shields.io/github/stars/Visual-Agent/DeepEyes.svg?style=social&label=Star)](https://github.com/Visual-Agent/DeepEyes)

+ [RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](https://arxiv.org/abs/2506.18369) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.18369)
  [![Star](https://img.shields.io/github/stars/oyt9306/RePIC.svg?style=social&label=Star)](https://github.com/oyt9306/RePIC)

+ [G1: Bootstrapping Perception and Reasoning Abilities of Vision-Language Model via Reinforcement Learning](https://arxiv.org/abs/2505.13426) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.13426)
  [![Star](https://img.shields.io/github/stars/chenllliang/G1.svg?style=social&label=Star)](https://github.com/chenllliang/G1)

#### Architectural \& Reasoning Mechanisms



+ [TACO: Think-Answer Consistency for Optimized Long-Chain Reasoning and Efficient Data Learning via Reinforcement Learning in LVLMs](https://arxiv.org/abs/2505.20777) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.20777)

+ [ProxyThinker: Test-Time Guidance through Small Visual Reasoners](https://arxiv.org/abs/2505.24872) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24872)
  [![Star](https://img.shields.io/github/stars/MrZilinXiao/ProxyThinker.svg?style=social&label=Star)](https://github.com/MrZilinXiao/ProxyThinker)

+ [GThinker: Towards General Multimodal Reasoning via Cue-Guided Rethinking](https://arxiv.org/abs/2506.01078) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01078)
  [![Star](https://img.shields.io/github/stars/jefferyZhan/GThinker.svg?style=social&label=Star)](https://github.com/jefferyZhan/GThinker)

+ [MiMo-VL Technical Report](https://arxiv.org/abs/2506.03569) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.03569)
  [![Star](https://img.shields.io/github/stars/XiaomiMiMo/MiMo-VL.svg?style=social&label=Star)](https://github.com/XiaomiMiMo/MiMo-VL)

+ [GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning](https://arxiv.org/pdf/2506.16141) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.16141)
  [![Star](https://img.shields.io/github/stars/TencentARC/GRPO-CARE.svg?style=social&label=Star)](https://github.com/TencentARC/GRPO-CARE)

+ [Machine Mental Imagery: Empower Multimodal Reasoning with Latent Visual Tokens](https://www.arxiv.org/abs/2506.17218) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.arxiv.org/abs/2506.17218)
  [![Star](https://img.shields.io/github/stars/UMass-Embodied-AGI/Mirage.svg?style=social&label=Star)](https://github.com/UMass-Embodied-AGI/Mirage)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vlm-mirage.github.io/)

+ [GRIT: Teaching MLLMs to Think with Images](https://arxiv.org/pdf/2505.15879) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.15879)

+ [SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](https://arxiv.org/abs/2506.01713) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.01713)
  [![Star](https://img.shields.io/github/stars/SUSTechBruce/SRPO_MLLMs.svg?style=social&label=Star)](https://github.com/SUSTechBruce/SRPO_MLLMs)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://srpo.pages.dev/#abstraction)

+ [VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](https://arxiv.org/pdf/2503.07523) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.07523)
  [![Star](https://img.shields.io/github/stars/zhangquanchen/VisRL.svg?style=social&label=Star)](https://github.com/zhangquanchen/VisRL)

+ [Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning](https://arxiv.org/abs/2505.12432) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.12432)
  [![Star](https://img.shields.io/github/stars/zrguo/Observe-R1.svg?style=social&label=Star)](https://github.com/zrguo/Observe-R1)


+ [VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning](https://arxiv.org/pdf/2505.22019) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.22019)
  [![Star](https://img.shields.io/github/stars/Alibaba-NLP/VRAG.svg?style=social&label=Star)](https://github.com/Alibaba-NLP/VRAG)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://modomodo-rl.github.io/)

+ [MoDoMoDo: Multi-Domain Data Mixtures for Multimodal LLM Reinforcement Learning](https://arxiv.org/abs/2505.24871) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24871)
  [![Star](https://img.shields.io/github/stars/lynl7130/MoDoMoDo.svg?style=social&label=Star)](https://github.com/lynl7130/MoDoMoDo)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://modomodo-rl.github.io/)


+ [Reason-RFT: Reinforcement Fine-Tuning for Visual Reasoning](https://arxiv.org/abs/2503.20752) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.20752)
  [![Star](https://img.shields.io/github/stars/tanhuajie/Reason-RFT.svg?style=social&label=Star)](https://github.com/tanhuajie/Reason-RFT)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://tanhuajie.github.io/ReasonRFT/)


+ [EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning](https://arxiv.org/pdf/2505.04623) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.04623)
  [![Star](https://img.shields.io/github/stars/HarryHsing/EchoInk.svg?style=social&label=Star)](https://github.com/HarryHsing/EchoInk/tree/main)


+ [One RL to See Them All: Visual Triple Unified Reinforcement Learning](https://arxiv.org/abs/2505.18129) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.18129)
  [![Star](https://img.shields.io/github/stars/MiniMax-AI/One-RL-to-See-Them-All.svg?style=social&label=Star)](https://github.com/MiniMax-AI/One-RL-to-See-Them-All)

+ [Enhancing LLMs' Reasoning-Intensive Multimedia Search Capabilities through Fine-Tuning and Reinforcement Learning](https://arxiv.org/abs/2505.18831) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.18831)

  

### Visual Agents with RL

+ [VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](https://arxiv.org/abs/2505.19255) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19255)
  [![Star](https://img.shields.io/github/stars/VTool-R1/VTool-R1.svg?style=social&label=Star)](https://github.com/VTool-R1/VTool-R1)


+ [VIKI‑R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://arxiv.org/abs/2506.09049) (June. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.09049)
  [![Star](https://img.shields.io/github/stars/MARS-EAI/VIKI-R.svg?style=social&label=Star)](https://github.com/MARS-EAI/VIKI-R)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://faceong.github.io/VIKI-R/)


+ [Enhancing Visual Grounding for GUI Agents via Self-Evolutionary Reinforcement Learning](https://arxiv.org/pdf/2505.12370) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.12370)
  [![Star](https://img.shields.io/github/stars/YXB-NKU/SE-GUI.svg?style=social&label=Star)](https://github.com/YXB-NKU/SE-GUI)


+ [VisualToolAgent (VisTA): A Reinforcement Learning Framework for Visual Tool Selection](https://arxiv.org/pdf/2505.20289) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20289)


+ [UI-R1: Enhancing Action Prediction of GUI Agents by Reinforcement Learning](https://arxiv.org/pdf/2305.15260) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.15260)
  [![Star](https://img.shields.io/github/stars/lll6gg/UI-R1.svg?style=social&label=Star)](https://github.com/lll6gg/UI-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://yxchai.com/UI-R1/)



### Visual World Models with RL
**Definition**: Learn predictive models of environment dynamics from visual inputs to enable planning and long-horizon reasoning in RL.


+ [Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2.pdf) (2025, Nature)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://www.nature.com/articles/s41586-025-08744-2.pdf)
  [![Star](https://img.shields.io/github/stars/danijar/dreamerv3.svg?style=social&label=Star)](https://github.com/danijar/dreamerv3)

+ [CoWorld: Making Offline RL Online: Collaborative World Models for Offline Visual Reinforcement Learning](https://arxiv.org/pdf/2305.15260) (2024, NeurIPS)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2305.15260)
  [![Star](https://img.shields.io/github/stars/qiwang067/CoWorld.svg?style=social&label=Star)](https://github.com/qiwang067/CoWorld)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://qiwang067.github.io/coworld)

+ [LS-Imagine: Open-World Reinforcement Learning over Long Short-Term Imagination](https://arxiv.org/abs/2410.03618) (2025, ICLR oral)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.03618)
  [![Star](https://img.shields.io/github/stars/qiwang067/LS-Imagine.svg?style=social&label=Star)](https://github.com/qiwang067/LS-Imagine)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://qiwang067.github.io/ls-imagine)


### Visual Generation with RL

**Definition**: Study RL agents that generate or manipulate visual content to achieve goals or enable creative visual tasks.

+ [Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization](https://arxiv.org/abs/2505.23331) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.23331)


+ [ReasonGen-R1: Cot for Autoregressive Image generation models through SFT and RL](https://arxiv.org/abs/2505.24875) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.24875)
  [![Star](https://img.shields.io/github/stars/Franklin-Zhang0/ReasonGen-R1.svg?style=social&label=Star)](https://github.com/Franklin-Zhang0/ReasonGen-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://reasongen-r1.github.io/)


+ [FocusDiff: Advancing Fine-Grained Text-Image Alignment for Autoregressive Visual Generation through RL](https://arxiv.org/pdf/2506.05501) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.05501)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://focusdiff.github.io/)


+ [DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models](https://arxiv.org/abs/2305.16381) (May. 2023, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.16381)
  [![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star)](https://github.com/google-research/google-research)


+ [A Simple and Effective Reinforcement Learning Method for Text-to-Image Diffusion Fine-tuning](https://arxiv.org/abs/2503.00897) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2503.00897)

+ [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) (May. 2023, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2305.13301)
  [![Star](https://img.shields.io/github/stars/jannerm/ddpo.svg?style=social&label=Star)](https://github.com/jannerm/ddpo)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://rl-diffusion.github.io/)


+ [PrefPaint: Aligning Image Inpainting Diffusion Model with Human Preference](https://arxiv.org/abs/2410.21966) (2024, Neurips)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.21966)
  [![Star](https://img.shields.io/github/stars/Kenkenzaii/PrefPaint.svg?style=social&label=Star)](https://github.com/Kenkenzaii/PrefPaint)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://prefpaint.github.io/)


+ [RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning](https://arxiv.org/abs/2505.17540) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.17540)
  [![Star](https://img.shields.io/github/stars/microsoft/DKI_LLM.svg?style=social&label=Star)](https://github.com/microsoft/DKI_LLM)


+ [Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](https://arxiv.org/abs/2407.06642) (2024, ECCV)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2407.06642)
  [![Star](https://img.shields.io/github/stars/wfanyue/DPG-T2I-Personalization.svg?style=social&label=Star)](https://github.com/wfanyue/DPG-T2I-Personalization)


+ [Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](https://arxiv.org/pdf/2311.15657) (2024, ECCV)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2311.15657)


+ [Rendering-Aware Reinforcement Learning for Vector Graphics Generation](https://arxiv.org/pdf/2505.20793) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.20793)


+ [GoT-R1: Unleashing Reasoning Capability of MLLM for Visual Generation with Reinforcement Learning](https://arxiv.org/abs/2505.17022) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.17022)
  [![Star](https://img.shields.io/github/stars/gogoduan/GoT-R1.svg?style=social&label=Star)](https://github.com/gogoduan/GoT-R1)


+ [DanceGRPO: Unleashing GRPO on Visual Generation](https://arxiv.org/abs/2505.07818) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.07818)
  [![Star](https://img.shields.io/github/stars/XueZeyue/DanceGRPO.svg?style=social&label=Star)](https://github.com/XueZeyue/DanceGRPO)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://dancegrpo.github.io/)


+ [LfVoid: Can Pre-Trained Text-to-Image Models Generate Visual Goals for Reinforcement Learning?](https://arxiv.org/abs/2307.07837) (2023, Neurips)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2307.07837)
  [![Star](https://img.shields.io/github/stars/gaojl19/LfVoid.svg?style=social&label=Star)](https://github.com/gaojl19/LfVoid)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://lfvoid-rl.github.io/)


### RL for Unified Model

+ [Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning](https://arxiv.org/abs/2505.03318) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.03318)
  [![Star](https://img.shields.io/github/stars/CodeGoat24/UnifiedReward.svg?style=social&label=Star)](https://github.com/CodeGoat24/UnifiedReward)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://codegoat24.github.io/UnifiedReward/think)


  
+ [MMaDA: Multimodal Large Diffusion Language Models](https://arxiv.org/abs/2505.15809) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.15809)
  [![Star](https://img.shields.io/github/stars/Gen-Verse/MMaDA.svg?style=social&label=Star)](https://github.com/Gen-Verse/MMaDA)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/spaces/Gen-Verse/MMaDA)


+ [VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning](https://arxiv.org/pdf/2504.02949) (Apr. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.02949)
  [![Star](https://img.shields.io/github/stars/VARGPT-family/VARGPT-v1.1.svg?style=social&label=Star)](https://github.com/VARGPT-family/VARGPT-v1.1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vargpt1-1.github.io/)


+ [Selftok: Discrete Visual Tokens of Autoregression, by Diffusion, and for Reasoning](https://arxiv.org/abs/2505.07538) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.07538)
  [![Star](https://img.shields.io/github/stars/selftok-team/SelftokTokenizer.svg?style=social&label=Star)](https://github.com/selftok-team/SelftokTokenizer)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://selftok-team.github.io/report/)



### RL for Embodied AI/Robotics

+ [VIKI‑R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](https://arxiv.org/pdf/2506.09049) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.09049)
  [![Star](https://img.shields.io/github/stars/MARS-EAI/VIKI-R.svg?style=social&label=Star)](https://github.com/MARS-EAI/VIKI-R)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://faceong.github.io/VIKI-R/)

+ [VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning](https://arxiv.org/pdf/2506.17221) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.17221)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vlnr1.github.io/)

+ [OctoNav: Towards Generalist Embodied Navigation](https://arxiv.org/pdf/2506.09839) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2506.09839)
  [![Star](https://img.shields.io/github/stars/buaa-colalab/OctoNav-R1.svg?style=social&label=Star)](https://github.com/buaa-colalab/OctoNav-R1)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://buaa-colalab.github.io/OctoNav/)

+ [Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](https://arxiv.org/abs/2506.00070) (May. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.00070)

+ [Embodied-R: Collaborative Framework for Activating Embodied Spatial Reasoning in Foundation Models via Reinforcement Learning](https://arxiv.org/pdf/2504.12680) (Apr. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2504.12680)

+ [Improving Vision-Language-Action Model with Online Reinforcement Learning](https://arxiv.org/pdf/2501.16664) (2025, ICRA)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2501.16664)

+ [RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning](https://arxiv.org/pdf/2412.09858) (Dec. 2024, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.09858)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://generalist-distillation.github.io/)

+ [Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2410.21845) (Oct. 2024, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.21845)
  [![Star](https://img.shields.io/github/stars/rail-berkeley/hil-serl.svg?style=social&label=Star)](https://github.com/rail-berkeley/hil-serl)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://hil-serl.github.io/)

+ [Integrating Reinforcement Learning with Foundation Models for Autonomous Robotics: Methods and Perspectives](https://arxiv.org/abs/2410.16411) (Oct. 2024, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2410.16411)
  [![Star](https://img.shields.io/github/stars/clmoro/Robotics-RL-FMs-Integration.svg?style=social&label=Star)](https://github.com/clmoro/Robotics-RL-FMs-Integration)

+ [MoDem: Accelerating Visual Model-Based Reinforcement Learning with Demonstrations](https://arxiv.org/abs/2212.05698) (Dec. 2022, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2212.05698)
  [![Star](https://img.shields.io/github/stars/facebookresearch/modem.svg?style=social&label=Star)](https://github.com/facebookresearch/modem)

+ [Selective Visual Representations Improve Convergence and Generalization for Embodied AI](https://openreview.net/pdf?id=kC5nZDU5zf) (2024, ICLR Spotlight)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://openreview.net/pdf?id=kC5nZDU5zf)
  [![Star](https://img.shields.io/github/stars/allenai/procthor-rl.svg?style=social&label=Star)](https://github.com/allenai/procthor-rl)

+ [Visual IRL for Human-Like Robotic Manipulation](https://arxiv.org/pdf/2412.11360) (Dec. 2024, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2412.11360)

+ [Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning](https://arxiv.org/abs/1803.09956) (2018, IROS)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/1803.09956)
  [![Star](https://img.shields.io/github/stars/andyzeng/visual-pushing-grasping.svg?style=social&label=Star)](https://github.com/andyzeng/visual-pushing-grasping)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://vpg.cs.princeton.edu/)


### RL for Medical Reasoning


+ [MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning](https://arxiv.org/pdf/2502.19634) (Feb. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2502.19634)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/JZPeterPan/MedVLM-R1)

### Audio Question Answering with RL

+ [Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering](https://arxiv.org/pdf/2503.11197) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2503.11197)
  [![Star](https://img.shields.io/github/stars/xiaomi-research/r1-aqa.svg?style=social&label=Star)](https://github.com/xiaomi-research/r1-aqa)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://huggingface.co/mispeech/r1-aqa)

### Others

#### Representation Learning

+ [Visual Pre-Training on Unlabeled Images using Reinforcement Learning](https://arxiv.org/abs/2506.11967) (Jun. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2506.11967)

+ [DINO-R1: Incentivizing Reasoning Capability in Vision Foundation Models](https://arxiv.org/pdf/2505.24025) (Mar. 2025, arXiv)
  [![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2505.24025)
  [![Website](https://img.shields.io/badge/Website-9cf)](https://christinepan881.github.io/DINO-R1/)

#### Blog

+ [Reinforcement Learning Guide](https://docs.unsloth.ai/basics/reinforcement-learning-guide) (2025, Blog)

+ [Can RL From Pixels be as Efficient as RL From State?](https://bair.berkeley.edu/blog/2020/07/19/curl-rad/) (Jul. 2025, Blog)

+ [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) (Mar. 2022, ICLR Blog)

#### Learning Course

+ [Deep Reinforcement Learning Course from Hugging Face](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction) (Hugging Face)

#### Survey

+ [Reinforced MLLM: A Survey on RL-Based Reasoning in Multimodal Large Language Models](https://arxiv.org/pdf/2504.21277) (May. 2025, arXiv)

+ [Reinforcement Learning for Generative AI: A Survey](https://arxiv.org/abs/2308.14328) (Aug. 2023, arXiv)

#### :high_brightness: This project is still on-going, pull requests are welcomed!!
If you have any suggestions (missing papers, new papers, or typos), please feel free to edit and submit a pull request. Even just suggesting paper titles is a great contribution — you can also open an issue or contact us via email.

#### :star: If you find this repo useful, please star it!!!


## Acknowledgements

This template is provided by [Awesome-Video-Diffusion](https://github.com/showlab/Awesome-Video-Diffusion). And our approach builds upon numerous contributions from prior resource, such as [Awesome Visual RL](https://github.com/qiwang067/awesome-visual-rl).

