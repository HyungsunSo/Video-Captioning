##Video-Captioning

소형선 Team Leader, Q-former architecture
강창우 Data Processing
김성진 Q-former architecture 
박준선 Perceiver architecture
홍경현 Perceiver architecture, 발표


graph TD
    subgraph "Input"
    A[Processed Video] 
    G[JSON Text]
    end

    A --> B[Pre-processing / Normalization]
    B --> C[Frame Extraction] 
    C --> D
    D --> E{Contrastive Learning}
    
    
    K --> E
    subgraph "CLiP Video"
        D[Frozen Image Encoder]
    end

    subgraph "CLiP Text"
        G --> K[Text Encoder - Text parsing]
    end
    
    subgraph "Adaptor Architecture"
        
        E -->|Query-Image Interaction| F[Learned Visual Tokens]
        
        
    end

    subgraph "LLM Generation"
        F --> H[Selective unfreezing LLM]
        H --> I[Generated Caption / Answer]
    end

    I --> J[Final Output]
