flowchart TD

    %% =========================
    %% Input
    %% =========================
    A["Input<br/>[B, 3, 544, 968] fp16"]

    %% =========================
    %% Pose model
    %% =========================
    subgraph POSE["Pose model"]
        A --> B["Pose backbone"]
        B --> C["stage_heatmaps<br/>[B, 19, 68, 121] fp16"]
        B --> D["stage_pafs<br/>[B, 38, 68, 121] fp16"]
    end

    %% =========================
    %% Adapter
    %% =========================
    subgraph ADAPTER["Adapter"]
        C --> E["Cast to fp32"]
        E --> F["Slice channels 0:18"]
        F --> G["heatmaps<br/>[B, 18, 68, 121] fp32"]

        D --> H["Cast to fp32"]
        H --> I["pafs<br/>[B, 38, 68, 121] fp32"]
    end

    %% =========================
    %% Heatmap branch
    %% =========================
    subgraph HEATMAP["Heatmap branch"]
        G --> J["Manual cubic resize<br/>[B, 18, 1080, 1920]"]
        J --> K["Separable NMS<br/>radius = 6<br/>[B, 18, 1080, 1920]"]
        K --> L["Peak mask<br/>hm == pooled AND hm > threshold<br/>[B, 18, 1080, 1920]"]
        L --> M["Where mask<br/>peaks ? hm : -1e9<br/>[B, 18, 1080, 1920]"]
        M --> N["Flatten spatial<br/>[B, 18, 2073600]"]
        N --> O["TopK<br/>K = 20<br/>axis = 2"]
        O --> P["top_scores<br/>[B, 18, 20] fp32"]
        O --> Q["top_indices<br/>[B, 18, 20] int64"]
    end

    %% =========================
    %% Coordinate decode
    %% =========================
    subgraph COORD["Coordinate decode"]
        Q --> R["Decode flat indices"]
        R --> S["x_all<br/>[B, 18, 20]"]
        R --> T["y_all<br/>[B, 18, 20]"]
    end

    %% =========================
    %% PAF pair scoring
    %% =========================
    subgraph PAFSCORING["PAF pair scoring"]
        S --> U["Select limb endpoint A/B coordinates"]
        T --> U
        P --> U

        U --> V["Pair grid<br/>[B, 19, 20, 20]"]
        V --> W["Line samples<br/>P = 8<br/>[B, 19, 20, 20, 8]"]

        I --> X["Sample PAFs<br/>from [B, 38, 68, 121]"]
        W --> X

        X --> Y["Dot product / valid points / affinity"]
        Y --> Z["pair_scores<br/>[B, 19, 20, 20]"]
        Y --> AA["pair_valid<br/>[B, 19, 20, 20]"]
    end

    %% =========================
    %% Pruning
    %% =========================
    subgraph PRUNING["Pruning"]
        Z --> AB["Reshape<br/>[B, 19, 400]"]
        AA --> AC["Mask invalid pairs"]
        AB --> AC
        AC --> AD["TopK<br/>M = 20<br/>axis = 2"]

        AD --> AE["limb_top_pair_score<br/>[B, 19, 20] fp32"]
        AD --> AF["flat_idx<br/>[B, 19, 20]"]

        AF --> AG["Decode flat pair index"]
        AG --> AH["limb_top_pair_a_idx<br/>[B, 19, 20] int64"]
        AG --> AI["limb_top_pair_b_idx<br/>[B, 19, 20] int64"]

        AE --> AJ["limb_top_pair_valid<br/>[B, 19, 20] fp32"]
    end

    %% =========================
    %% MXR outputs to CPU
    %% =========================
    subgraph CPUOUT["MXR outputs sent to CPU"]
        P --> OUT1["top_scores<br/>[B, 18, 20]"]
        Q --> OUT2["top_indices<br/>[B, 18, 20]"]
        AH --> OUT3["limb_top_pair_a_idx<br/>[B, 19, 20]"]
        AI --> OUT4["limb_top_pair_b_idx<br/>[B, 19, 20]"]
        AE --> OUT5["limb_top_pair_score<br/>[B, 19, 20]"]
        AJ --> OUT6["limb_top_pair_valid<br/>[B, 19, 20]"]

        OUT1 --> CPU["CPU pose assembly"]
        OUT2 --> CPU
        OUT3 --> CPU
        OUT4 --> CPU
        OUT5 --> CPU
        OUT6 --> CPU
    end

    %% =========================
    %% Highlight expensive full-res path
    %% =========================
    classDef bottleneck fill:#ffe0e0,stroke:#cc0000,stroke-width:2px;
    class J,K,L,M,N bottleneck;