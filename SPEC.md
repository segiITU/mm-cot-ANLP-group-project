# mm-cot 

This is a project that investigates how masking different components of chain-of-thought reasoning affects model performance in multimodal question answering, specifically the mm-cot arhitecture by Amazon Science.

This ia an overview of the Multimodal-CoT framework: Multimodal-CoT consists of two stages: (i) rationale generation and (ii) answer inference. Both stages share the same model structure but differ in the input and output. In the first stage, we feed the model with language and vision inputs to generate rationales. In the second stage, we append the original language input with the rationale generated from the first stage. Then, we feed the updated language input with the original vision input to the model to infer the answer.

Our goal is to create a repository that does the following: 

- runs a single ScienceQA question and image through the two-stage mm-cot pipeline 
- adding masking effects on the rationale generation

The name of the project is 'mm-cot-ANLP-group-project' and uses Git for source control. 