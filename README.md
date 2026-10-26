# AA-5750: Contemporary Issues in Analytics

**From Foundational Machine Learning to Modern AI Systems**

This repository contains course materials for **AA-5750: Contemporary Issues in Analytics** at Saint Louis University (SLU).

The course provides a practical, architecture-driven exploration of contemporary artificial intelligence. Students progress from data augmentation and convolutional neural networks (CNNs) to residual networks (ResNets), sequence modeling, transformer-based language models, AI agents, and retrieval-augmented generation (RAG).

Rather than treating AI models as black boxes, the course emphasizes understanding their underlying principles, computational mechanisms, architectural design choices, implementation, and practical applications through hands-on learning and experimentation.

## Course Roadmap

The course follows an eight-week progression, connecting foundational machine learning techniques with modern deep learning architectures and AI systems.

| Week | Topic | Key Concepts |
|---|---|---|
| 1 | Data Augmentation | Data preprocessing, image and text augmentation, dataset diversity, generalization |
| 2 | Convolutional Neural Networks (CNNs) | Convolution, filters, feature maps, pooling, receptive fields, image classification |
| 3 | Residual Networks (ResNets) | Residual blocks, skip connections, identity mappings, deep network optimization |
| 4 | ResNet: Advanced Exploration | Architectural analysis, training experiments, convergence, performance evaluation, error analysis |
| 5 | Sequence Modeling: LSTMs and GRUs | Recurrent neural networks, hidden states, gating mechanisms, sequential dependencies |
| 6 | Transformers and BERT | Attention, self-attention, positional encoding, transformer architecture, contextual representations |
| 7 | AI Agents | Agent architectures, tool use, planning, multi-step workflows, task execution |
| 8 | Retrieval-Augmented Generation (RAG) | Embeddings, document retrieval, vector search, retrieval pipelines, grounded generation |

## Weekly Learning Modules

### Week 1 — Data Augmentation

Explore how data augmentation increases training-data diversity and helps machine learning models generalize beyond their training examples.

**Key areas:**
- Data foundations and preprocessing
- Image and text augmentation techniques
- Image transformations and augmentation strategies
- Dataset size, diversity, and representation
- Training and validation considerations
- Evaluating augmentation's impact on model performance
- Understanding the relationship between augmentation and generalization

### Week 2 — Convolutional Neural Networks (CNNs)

Study the architecture and computational principles of convolutional neural networks, focusing on how convolutional filters extract spatial features and build hierarchical visual representations.

**Key areas:**
- CNN architecture and its components
- Convolution operations, kernels, and filters
- Stride, padding, and output dimensions
- Feature maps and receptive fields
- Activation functions and nonlinear feature learning
- Pooling and spatial downsampling
- Parameter sharing and local connectivity
- CNN training and image classification
- Visualizing and interpreting learned features
- Evaluating model performance and generalization

### Week 3 — Residual Networks (ResNets)

Explore how residual learning extends convolutional neural networks to support the training of deeper architectures and address optimization challenges associated with increasing network depth.

**Key areas:**
- Review of CNN building blocks
- Limitations of plain deep convolutional networks
- Residual functions and residual blocks
- Skip connections and identity mappings
- Gradient propagation and optimization
- ResNet architecture and its components
- Image classification experiments
- Comparing CNN and ResNet performance
- Understanding the benefits and limitations of residual learning

### Week 4 — ResNet: Advanced Exploration

Extend the study of residual networks through architectural analysis, controlled experiments, and empirical evaluation. Investigate how architectural configurations and training choices influence convergence, predictive performance, and computational cost.

**Key areas:**
- Residual network architecture and components
- Basic blocks and bottleneck blocks
- ResNet variants and architectural configurations
- Training behavior, optimization, and convergence
- Model configuration and controlled experimentation
- Hyperparameters and training strategies
- Performance evaluation and error analysis
- Computational efficiency and architectural trade-offs
- Interpreting experimental results and limitations

### Week 5 — Sequence Modeling: LSTMs and GRUs

Explore recurrent architectures designed to process sequential data and model temporal dependencies. Examine how gating mechanisms help recurrent networks manage information over time.

**Key areas:**
- Sequential data and recurrent neural networks
- Hidden states and recurrent computation
- Sequence representations and temporal dependencies
- Vanishing and exploding gradients
- Long short-term memory (LSTM) networks
- Gated recurrent units (GRUs)
- Input, forget, and output gates
- Comparing recurrent architectures
- Training and evaluating sequence models

### Week 6 — Transformers and BERT

Examine the transition from recurrent sequence modeling to attention-based architectures. Study how transformers model relationships between tokens and how BERT learns contextual language representations.

**Key areas:**
- Limitations of recurrent sequence models
- Attention and self-attention
- Query, key, and value representations
- Scaled dot-product attention
- Multi-head attention
- Positional encoding and positional information
- Transformer encoder architecture
- Bidirectional Encoder Representations from Transformers (BERT)
- Pretraining and contextual representations
- Fine-tuning pretrained language models
- Evaluating transformer-based models

### Week 7 — AI Agents

Explore AI systems that extend beyond individual model calls by coordinating language models, tools, and multi-step workflows to accomplish tasks.

**Key areas:**
- Agent architecture and core components
- Language models as agent controllers
- Tool calling and external integrations
- Planning and task decomposition
- Multi-step execution and feedback
- State management and workflow orchestration
- Handling errors and failed tool calls
- Agent evaluation, reliability, and limitations
- Practical agent workflow implementation

### Week 8 — Retrieval-Augmented Generation (RAG)

Study how retrieval systems connect external knowledge sources to generative language models to support context-aware, evidence-grounded responses.

**Key areas:**
- Limitations of standalone language models
- Document ingestion and preprocessing
- Document chunking and context construction
- Text embeddings and semantic representations
- Vector search and similarity measures
- Retrieval pipelines and candidate selection
- Connecting retrieved context to language models
- Retrieval-augmented generation workflows
- Evaluating retrieval quality and response grounding
- Common failure modes and system limitations

## Learning Approach

The course connects conceptual understanding with practical implementation, architectural analysis, and empirical evaluation.

Students are encouraged to:

1. **Understand the foundations:** Learn the motivation behind each algorithm and architecture, including the problems it is designed to solve.
2. **Examine the architecture:** Identify the major components, understand their computational roles, and investigate how they interact.
3. **Implement and experiment:** Apply concepts through programming exercises, model implementations, and practical demonstrations.
4. **Evaluate performance:** Analyze experimental results, training behavior, model predictions, and limitations using appropriate evaluation metrics.
5. **Compare design choices:** Investigate how architectural and training decisions influence performance, efficiency, and generalization.
6. **Connect ideas across weeks:** Understand how advances in neural networks, convolutional architectures, residual learning, sequence modeling, attention, and retrieval contribute to modern AI systems.

## Repository Structure

```text
AA-5750-Contemporary_Issues_In_Analytics/
│
├── AA-5750 Week-1 2/    # Data Augmentation
├── AA-5750 Week-2/      # Convolutional Neural Networks (CNNs)
├── AA-5750 Week-3/      # Residual Networks (ResNets)
├── AA-5750 Week-4/      # ResNet: Advanced Exploration
├── AA-5750 Week-5/      # LSTMs and GRUs
├── AA-5750 Week-6/      # Transformers and BERT
├── AA-5750 Week-7/      # AI Agents
├── AA-5750 Week-8/      # Retrieval-Augmented Generation (RAG)
│
└── README.md
```

*Folder names and contents may evolve as course materials are organized and updated.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ChandraPrakash-Bathula/AA-5750-Contemporary_Issues_In_Analytics.git
```

### 2. Navigate to the repository

```bash
cd AA-5750-Contemporary_Issues_In_Analytics
```

### 3. Explore the weekly modules

Open the directory corresponding to the relevant week and follow the instructions provided in its course materials.

### 4. Configure the environment

Refer to each module's instructions for its specific dependencies, environment configuration, datasets, and execution steps. Requirements may differ across modules depending on the frameworks and tools used.

## Intended Audience

This repository is intended primarily for **Saint Louis University (SLU) students enrolled in AA-5750: Contemporary Issues in Analytics**.

The course materials support students in:

- Understanding foundational machine learning and deep learning concepts.
- Exploring CNNs and residual architectures through implementation and experimentation.
- Developing an understanding of sequential models, transformers, and pretrained language models.
- Building practical skills in AI agents and retrieval-augmented generation.
- Connecting theoretical principles with real-world AI applications.
- Developing the ability to analyze model architectures, evaluate experimental results, and identify system limitations.
- Establishing a foundation for further study, applied research, and projects in contemporary artificial intelligence.

## Instructor

**Chandra Prakash Bathula**  
Adjunct Faculty, Saint Louis University

- [GitHub](https://github.com/ChandraPrakash-Bathula)
- [LinkedIn](https://www.linkedin.com/in/chandra-prakash-bathula/)
- [Medium](https://medium.com/@ChandraPrakash-Bathula)

## Academic Use

These materials are intended for educational purposes. Students should follow applicable course instructions, assignment requirements, dataset licensing terms, and academic integrity guidelines when completing coursework or adapting the materials.

## Course Summary

**AA-5750: Contemporary Issues in Analytics** connects foundational machine learning concepts with contemporary deep learning architectures and AI applications. Through an eight-week progression, students explore how models are designed, implemented, trained, evaluated, and integrated into larger AI systems.

The course emphasizes not only using AI technologies, but also developing a deeper understanding of the architectural principles and computational mechanisms that make them work.

---

**AA-5750: Contemporary Issues in Analytics**  
*Connecting foundational algorithms, deep learning architectures, and modern AI applications.*
