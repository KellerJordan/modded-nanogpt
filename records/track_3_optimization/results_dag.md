# Track 3 Results Dependency DAG

Solid edges are dependencies stated directly in the README, such as "Setup from #20" or "following #8".
Dotted edges are lighter inferred lineage links for variants of earlier result families.
Nodes labeled `PR #...` are open Track 3 pull requests that have not been accepted into the README results history yet.

Open Track 3 PRs included here:
[PR #286](https://github.com/KellerJordan/modded-nanogpt/pull/286),
[PR #295](https://github.com/KellerJordan/modded-nanogpt/pull/295),
[PR #297](https://github.com/KellerJordan/modded-nanogpt/pull/297),
[PR #307](https://github.com/KellerJordan/modded-nanogpt/pull/307),
[PR #308](https://github.com/KellerJordan/modded-nanogpt/pull/308),
[PR #309](https://github.com/KellerJordan/modded-nanogpt/pull/309),
[PR #310](https://github.com/KellerJordan/modded-nanogpt/pull/310),
[PR #311](https://github.com/KellerJordan/modded-nanogpt/pull/311),
[PR #312](https://github.com/KellerJordan/modded-nanogpt/pull/312),
and [PR #318](https://github.com/KellerJordan/modded-nanogpt/pull/318).

```mermaid
flowchart LR
    R1["#1 Muon"]
    R2["#2 AdamW"]
    R3["#3 Muon hparams"]
    R4["#4 AdamH"]
    R5["#5 MuonH"]
    R6["#6 Muon wd"]
    R7["#7 Muon2"]
    R8["#8 NorMuonH"]
    R9["#9 NorMuon + u/w floor"]
    R10["#10 NorMuon"]
    R11["#11 Contra-Muon"]
    R12["#12 Muon WSD"]
    R13["#13 MuLoCo NorMuonH"]
    R14["#14 SOAP-Muon MLP"]
    R15["#15 Newton-Muon"]
    R16["#16 SOAP attention trust"]
    R17["#17 Aurora"]
    R18["#18 PMuon"]
    R19["#19 KL-SOAP-H"]
    R20["#20 Contra + Soft-Muon"]
    R21["#21 Shampoo"]
    R22["#22 Spectral Descent"]
    R23["#23 Muown"]
    R24["#24 Split cooldown"]
    R25["#25 KL-SOAP power LR"]
    R26["#26 SinkSOAP"]
    R27["#27 SOAPH"]
    R28["#28 DynMuon"]
    R29["#29 Radial brake"]
    R30["#30 Aurora + extended Contra"]
    R31["#31 ContraNormMuown"]
    R32["#32 SODA fade"]
    R33["#33 PSGD"]
    R34["#34 capped RRE"]
    P286["PR #286 Aurora + SOAP-MLP"]
    P295["PR #295 Normalized Correction"]
    P297["PR #297 Exact Frob Init + tangent u/w"]
    P307["PR #307 tail reference interpolation"]
    P308["PR #308 EMA-Nesterov + Muon"]
    P309["PR #309 EMA-Nesterov + Aurora"]
    P310["PR #310 Arbor Muon"]
    P311["PR #311 Circuit-Muon"]
    P312["PR #312 Aurora EMA reference"]
    P318["PR #318 tail trajectory updates"]

    %% Inferred early lineage / optimizer-family variants.
    R1 -.-> R3
    R1 -.-> R6
    R1 -.-> R7
    R1 -.-> R9
    R1 -.-> R10
    R1 -.-> R15
    R1 -.-> R18
    R1 -.-> R22
    R1 -.-> R23
    R1 -.-> R28
    R2 -.-> R4
    R4 -.-> R5
    R5 -.-> R8
    R10 -.-> R26
    R21 -.-> R22

    %% Explicit README ancestry.
    R9 --> R11
    R10 --> R12
    R8 --> R12
    R8 --> R13
    R11 --> R14
    R14 --> R16
    R11 --> R17
    R8 --> R19
    R16 --> R20
    R11 --> R24
    R19 --> R25
    R19 --> R27
    R20 --> R29
    R29 --> R30
    R23 --> R31
    R20 --> R32
    R30 --> R34

    %% Open, not-yet-accepted Track 3 PRs.
    R11 --> P286
    R14 -.-> P286
    R17 -.-> P286
    R6 --> P295
    R12 -.-> P295
    R5 -.-> P295
    R29 --> P297
    R30 --> P307
    R12 --> P308
    R30 --> P309
    R12 --> P310
    P309 --> P311
    P309 --> P312
    P311 --> P318

    classDef record fill:#f7f7f7,stroke:#444,stroke-width:1px,color:#111;
    classDef pending fill:#fff7df,stroke:#b87900,stroke-width:1px,color:#111;
    class R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11,R12,R13,R14,R15,R16,R17,R18,R19,R20,R21,R22,R23,R24,R25,R26,R27,R28,R29,R30,R31,R32,R33,R34 record;
    class P286,P295,P297,P307,P308,P309,P310,P311,P312,P318 pending;
```
