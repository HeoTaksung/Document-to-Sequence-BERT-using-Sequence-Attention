# Document-to-Sequence-BERT

Multi-Label Classification

  * [Medical Code Prediction from Discharge Summary: Document to Sequence BERT using Sequence Attention](https://arxiv.org/abs/2106.07932)
  
    * `Tak-Sung Heo`, `Yongmin Yoo`, `Yeongjoon Park`, `Byeong-Cheol Jo`

-------------------------------------------------

## Dataset

  * [MIMIC-III](https://mimic.mit.edu/iv/)
    
    * Text - noteevents
    
    * Label - diagnoses_icd, procedure_icd

-------------------------------------------------

## Model Structure

  * Document to Sequence Preprocessing

  * Collecting the CLS_token extracted through Document-to-Sequence BERT (D2SBERT)
    
    * We used [BioBERT](https://github.com/dmis-lab/biobert)
    
  * Sequence Attention
  
  * Classifier
  
 -------------------------------------------------
 
 ## Result
 
  |    Model    | F1-Macro  | F1-Micro  |
  | :------: | :---: | :-----: |
  |  [CAML](https://www.aclweb.org/anthology/N18-1100.pdf)               | 0.56924      | 0.64993      |
  |  [SWAM](https://arxiv.org/pdf/2101.11430.pdf)               | 0.58025      | 0.65994      |
  |  [EnCAML](https://www.sciencedirect.com/science/article/pii/S0167739X21000236?casa_token=jeJOkYcrI_AAAAAA:OK9kI-9P3BFEeGOjWygNlDKsJlzclMZeDsJ0rhRP7Mvdcrb5nxGpGGhl7ewRBi5cKQXRWxJWLQ)             | 0.59653      | 0.66594      |
  |  [BERT-head](https://link.springer.com/chapter/10.1007/978-3-030-32381-3_16)          | 0.49376      | 0.56627      |
  |  [BERT-tail](https://link.springer.com/chapter/10.1007/978-3-030-32381-3_16)          | 0.45453      | 0.54011      |
  |  [BERT-head-tail](https://link.springer.com/chapter/10.1007/978-3-030-32381-3_16)     | 0.49362      | 0.56566      |
  |  **Proposed model** | **0.62898**  | **0.68555**  |
