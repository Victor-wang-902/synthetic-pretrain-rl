# Pre-training with Synthetic Data Helps Offline Reinforcement Learning
Official code for our paper [Pre-training with Synthetic Data Helps Offline Reinforcement Learning](https://openreview.net/forum?id=PcxQgtHGj2&referrer=%5Bthe%20profile%20of%20Che%20Wang%5D(%2Fprofile%3Fid%3D~Che_Wang1)) at ICLR 2024.


# Installation
Our environment setup should be largely similar to our prior work. Refer to `environment.yml` for more information. **Note**: To directly use our script for pre-training the Decision Transformer, you would need to install [HuggingFace Accelerate](https://github.com/huggingface/accelerate) and set up a *Duo-GPU* configuration. You can find tutorials for Accelerate [here](https://huggingface.co/docs/accelerate/en/index). With our settings, we were able to pre-train the model for 80,000 steps with 2 RTX 8000 GPUs within 18 hours.


# Data Generation
To generate synthetic data, run the example commands in `syn_pretraining/run_generate.sh`. You might have to play around with `num_workers` argument based on your CPU availability.


# Pre-training Decision Transformer
After obtaining the dataset, run the example commands in `syn_pretraining/run_pretrain.sh`.

# Pre-training CQL
The code in `syn_pretraining_cql/pretrain_cql.py` is self-contained with synthetic data MDP generation and pre-training MLP Q networks which can be used for subsequent CQL fine-tuning. As mentioned in our paper, the code for CQL experiments is adapted from [this acknowledged CQL repository](https://github.com/young-geng/CQL/tree/master). Therefore, the implementation of Q networks (critics) also follows their overall structure in this pre-training code. To run CQL pre-training experiments, please refer to `syn_pretraining_cql/pretrain_cql.sh` for exemplary command-line executions. 


# Updates
May 25: As of now, we are still cleaning our code. As per many have request, we will prioritize on releasing the code for the synthetic generation/pre-training, as it is the central part of this project. Please stay tuned for more updates!

May 26: Released the code for synthetic data generation and pre-training with Decision Transformer (DT)

May 27: Released the code for synthetic MDP data generation and pre-training with Conservative Q-Learning (CQL)
