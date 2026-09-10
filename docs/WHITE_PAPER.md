# Project N: A Dyadic, Multimodal, and Transactional Framework for Communication Support in Completely Non-Verbal Autism

**Lead Researcher & Project Architect:** Valentyn Shybanov ([olostan.me](https://olostan.me/) · [GitHub](https://github.com/olostan))  
**Collaborative Scope:** Speech-Language Pathologists (SLPs), Occupational Therapists (OTs), Developmental Pediatricians, Neurodiversity Researchers, and Assistive Technology Practitioners  
**Document Classification:** Scientific Whitepaper & Clinical Foundations  
**Version:** 1.1.0 — September 2026  

---

## Foreword from the Founder

> *"I am a software engineer, systems architect, and—most importantly—the father of Nolan, a 7-year-old completely non-verbal boy with Level 3 autism. Every single day of my life is shaped by the profound love, challenge, and heartbreak of trying to understand my child. Nolan is non-verbal. Completely.*
>
> *My son cannot speak in words, but he is never silent. He communicates continuously: through subtle pitch inflections in his throat, micro-tremors in his hands, bodily orientations, and rhythms of movement. Traditional society and standard AI models discard these signals as 'meaningless noise.' But to me, as his father, that 'noise' is his entire voice.*
>
> *I did not start Project N as a commercial startup. I have no desire to monetize this software, sell a proprietary device, or exploit vulnerable families. I started this project because I refuse to accept that my son must remain trapped behind a barrier of communication. I am dedicating my twenty-plus years of engineering experience, machine learning knowledge, and systems architecture skills to build an open-source, local-first, privacy-respecting tool that can help parents like me and the dedicated therapists who support our children.*
>
> *This whitepaper outlines the scientific, clinical, and architectural foundation of this initiative. If you are a speech-language pathologist, an occupational therapist, an autism researcher, or an engineer who believes in communication rights: I warmly invite you to read, critique, and contribute to this open endeavor."*
>
> — **Valentyn Shybanov** ([olostan.me](https://olostan.me/))

---

## Abstract

Completely non-verbal autistic individuals (such as children requiring Level 3 supports) communicate through an intricate, non-phonetic multimodal repertoire: idiosyncratic vocalizations, continuous tonal hums, micro-pitch inflections, repetitive kinetic motor stims, postural adjustments, and physiological shifts. Traditional artificial intelligence paradigms attempt to process non-verbal communication through an extractive "intent translation" lens—treating the child as an isolated, closed-box signal generator and attempting to decode an internal mental state into neurotypical English text. 

This paper presents the scientific rationale and methodological foundation for **Project N**, a local-first, privacy-preserving computational framework that departs fundamentally from extractive decoding. Grounded in the **Transactional Model of Communication** ([Sameroff, 1975](#ref-16); [Wetherby & Prizant, 2000](#ref-20)) and the **SCERTS Framework** ([Prizant et al., 2006](#ref-14)), Project N conceptualizes non-verbal communication as an active, **dyadic co-regulatory loop** between the child and their communicative partners (caregivers, therapists, educators). Rather than relying on acoustic features alone or optimizing on retrospective adult labels—which risks manufacturing confirmation bias and Facilitated Communication (FC) epistemic traps ([National Autism Center, 2026](#ref-13))—Project N anchors its truth criterion in **prospective Behavioral Resolution** (verifying whether an offered co-regulatory support objectively resolves distress and restores baseline within an observed latency window) augmented by direct, optional child-authored AAC selection. By combining bioacoustics, computer vision kinematics, and episodic memory, the framework provides clinicians, occupational therapists, and families with an evidence-grounded, four-layer decision-support assistant that honors child agency and preserves scientific integrity.

---

## 1. Visual Overview: The Dyadic Transactional Architecture

### 1.1 The Dyadic Co-Regulatory Sequence
```mermaid
sequenceDiagram
    autonumber
    actor Child as Child N (Completely Non-Verbal)
    actor Partner as Communication Partner (Parent / OT / SLP)
    participant Engine as Project N Local Assistant (Apple Silicon)
    participant AAC as Child's AAC Speech Device (Optional)

    Child->>Partner: Non-verbal bid (Pitch glide + 4Hz wrist stim)
    Note over Partner: Partner observes context & captures clip via mobile client
    Partner->>Engine: Uploads clip with situational antecedents
    Engine->>Engine: Extracts holistic acoustics (F0, CQT) & 3D kinematics (pose, optical flow)
    Engine->>Engine: Evaluates Medical Safety Gate (NCCPC-PV triage first)
    Engine->>Engine: Retrieves matching historical episodes & clinical research precedents
    Engine->>Partner: Delivers Caregiver & Therapist Insight Card:<br/>• Acoustic strain & motion analysis<br/>• Historical co-regulatory matches (e.g., deep pressure resolved 2/3)<br/>• Grounded OT/SLP recommendations & hints
    opt Optional Child Authorship
        Engine->>AAC: Pre-populates candidate tiles ([Deep Pressure], [Sensory Break])
        Child->>AAC: Directly selects icon or gestures
        AAC-->>Partner: Speaks aloud child's choice
    end
    Partner->>Child: Delivers targeted co-regulatory support (Deep pressure / quiet space)
    Note over Child,Partner: Co-regulation restored; latency & outcome recorded
    Engine->>Engine: Updates local N-of-1 episodic memory with verified outcome
```

### 1.2 The SCERTS Framework & Multimodal Integration
```mermaid
graph TD
    subgraph SCERTS Foundation [Clinical & Developmental Pillars]
        SC[Social Communication: Spontaneous Functional Bids]
        ER[Emotional Regulation: Arousal & Sensory Homeostasis]
        TS[Transactional Support: Partner Scaffolding & Visual AAC]
    end

    subgraph Sensory Observables [Child Multimodal Repertoire]
        Acoustic[Acoustic Physics: Micro-Pitch F0, CQT Harmonics, Strain CPP]
        Kinematic[Body-Relative Kinematics: 75 Pose Keypoints, Optical Flow]
        Physio[Autonomic Physiology: EDA Conductance, HRV Vagal Tone]
    end

    subgraph Partner Dynamics [Communication Partner Context]
        AdultWords[Adult Verbal Scaffolding & Prompts]
        AdultActions[Offered Supports: Pressure, Water, Visual Schedule]
        Environment[Antecedents: Transitions, Ambient Noise, Meals]
    end

    SC & ER & TS --> Acoustic & Kinematic & Physio
    AdultWords & AdultActions & Environment --> JointEvent[Joint Transactional Event]
    Acoustic & Kinematic & Physio --> JointEvent

    subgraph Analytical Core [Project N Local Processing]
        JointEvent --> SafetyGate{NCCPC-PV Pain Score >= 6<br/>or Sensory Distress Alert?}
        SafetyGate -->|Yes| MedicalCard[Medical Red-Flag Escalation Card]
        SafetyGate -->|No| MetricHead[128-dim L2 Metric Projection Head]
        MetricHead --> EpisodicMemory[(Historical Verified Precedents)]
        EpisodicMemory --> RAG[(Clinical Evidence Library: FBA, HIPPEA, Interoception)]
    end

    subgraph Empowerment [Caregiver & Therapist Guidance with Child Agency]
        RAG --> CaregiverCard[Primary: Four-Layer Structured Evidence Card L1-L4<br/>• Behavioral Analysis & Pattern Recognition<br/>• What Helped Before & Practical Hints for OTs/Parents<br/>• Grounded Clinical Literature Citations]
        RAG --> AACBridge[Optional Child Bridge: Adaptive AAC Choice Tiles<br/>• Direct Child Self-Advocacy & Authorship<br/>• Real-time Visual Choice Board]
    end

    style MedicalCard fill:#ff4d4f,color:#fff,stroke:#333,stroke-width:2px
    style AACBridge fill:#52c41a,color:#fff,stroke:#333,stroke-width:2px
    style CaregiverCard fill:#1890ff,color:#fff,stroke:#333,stroke-width:2px
```

---

## 2. Introduction & The Communicative Reality

### 2.1 Beyond the Verbal Threshold: Non-Speaking Is Not Non-Communicative
A substantial proportion (estimated at 25% to 30%) of autistic children remain minimally speaking or non-verbal past school age ([Tager-Flusberg & Kasari, 2013](#ref-18)). For these individuals, the absence of functional speech does not signify a lack of communicative intent, cognitive agency, or receptive language comprehension. Rather, co-occurring challenges—including Childhood Apraxia of Speech (CAS), oral-motor dyspraxia, sensory processing differences, and atypical sensorimotor feedforward integration—disrupt the complex neuromuscular coordination required to articulate discrete phonemic sequences.

In clinical Speech-Language Pathology (SLP), communication is recognized as inherently multimodal ([Light & McNaughton, 2014](#ref-8)). Non-verbal children routinely mobilize an extensive communicative repertoire:

- **Paralinguistic Acoustic Cues:** Continuous vocal fold vibrations, glottal stops, clicks, harmonic overtone sweeps, guttural resonance, and subtle micro-pitch inflections (hypothesized in this N-of-1 deployment to manifest within $\pm 15\text{ to } 50\text{ Hz}$ excursions) that reflect physiological equilibrium, affective valence, or protest.
- **Kinematic & Motor Dynamics:** High-frequency repetitive motor behaviors ("stims"), including 3–6 Hz wrist rotations, finger-flicking in peripheral vision, pacing, torso rocking, or intentional physical reaches.
- **Physiological & Autonomic Fluctuations:** Sympathetic nervous system arousal, electrodermal reactivity, and cardiorespiratory shifts (HRV) driven by sensory demands.

### 2.2 The Neurotypical Inductive Bias of Commercial AI
Conventional foundation models (speech-to-text engines like Whisper, audio transformers like AST, and vision transformers like CLIP/ViT) are trained on massive datasets of neurotypical human speech and cinematic macro-actions. Their loss functions are mathematically optimized to enforce **phonemic and spatial collapse**:

1. **Acoustic Phonemic Discretization:** Whisper’s autoregressive decoder penalizes acoustic variance that does not map onto standardized phonemes or linguistic tokens. Idiosyncratic tonal hums or guttural phonations are filtered out as background noise or collapsed into silence.
2. **Visual Spatial Pooling:** Standard vision models downsample spatial patches across temporal windows, blending a rapid 4 Hz wrist-flick or finger tremor into the static background pixels of a living room.

When applied to a completely non-verbal child, commercial AI obliterates the very substrate that constitutes their expressive vocabulary.

---

## 3. The Transactional Paradigm: Communication as a Dyadic Loop

### 3.1 The Limits of "Unidirectional Intent Decoding"
Early artificial intelligence approaches to non-verbal autism conceptualized the child as a standalone transmitter whose "signals" must be translated by an algorithmic receiver. From a clinical speech therapy perspective, this assumption is fundamentally flawed:

In naturalistic pediatric communication, a vocalization or movement does not have an invariant, one-to-one semantic translation. A 350 Hz vocalization paired with hand flapping may represent intense joy during water play, severe vestibular seeking during room transitions, or overwhelming autonomic distress when ambient noise exceeds threshold. Identical surface behaviors arise from divergent internal needs, while a single functional need can manifest through varied behavioral expressions (by analogy with the context-dependent emotion-inference critique of [Barrett et al., 2019](#ref-1)).

### 3.2 The SCERTS Framework & Interpersonal Scaffolding
To achieve clinical validity, Project N is architected around the **SCERTS Model** ([Prizant et al., 2006](#ref-14)), an internationally recognized, evidence-based multidisciplinary framework focusing on:
- **SC (Social Communication):** Developing spontaneous, functional communication and emotional expression across non-verbal and aided modalities.
- **ER (Emotional Regulation):** Supporting self-regulation and mutual regulation to maintain optimal arousal states for learning and interaction.
- **TS (Transactional Support):** The interpersonal supports, environmental modifications, and learning tools provided by communication partners.

### 3.3 Why Labeling Adult Interaction Is Vital
A central insight of speech-language pathology is that a non-speaking child’s communication is shaped by their communicative partner's behavior ([Sameroff, 1975](#ref-16); [McLean & Snyder-McLean, 1978](#ref-9)). 

Project N incorporates adult interaction data into its core relational schema:
- **What did the adult say or ask?** (e.g., *"Are you hungry?"*, *"Let's go outside"*, verbal pause/wait time).
- **What physical support was offered?** (e.g., offering a visual choice board, providing deep proprioceptive pressure, opening a door, reducing sensory input).
- **How did the child respond to that specific interaction?** (e.g., physical reach, turning away, vocal pitch lowering, respiration settling, or distress escalation).
- **Latency to Resolution:** The elapsed time required for the child to return to a regulated baseline following the adult's interaction.

By modeling the **dyad** rather than the child in isolation, the system learns which adult transactional supports correlate with successful co-regulation and communicative connection.

---

## 4. Regulatory State Progression & Medical Safeguards

```mermaid
stateDiagram-v2
    [*] --> BaselineRegulated: Homeostatic Equilibrium

    BaselineRegulated --> SensorySeeking: High Prediction Error / Hypo-Arousal
    SensorySeeking --> BaselineRegulated: Stimming / Proprioceptive Feedback Restores Equilibrium

    BaselineRegulated --> Dysregulation: Environmental Overload / Transition / Fatigue
    Dysregulation --> MedicalTriage: Observable Distress Indicators Present

    state MedicalTriage {
        [*] --> CheckNCCPC: Evaluate 27-Item NCCPC-PV Checklist
        CheckNCCPC --> RedFlagEscalation: Score >= 6 (Physical Pain Suspected)
        CheckNCCPC --> BehavioralDyad: Score < 6 (Sensory / Communicative)
    }

    RedFlagEscalation --> ClinicalMedicalReview: Alert Caregiver to Examine Physical Cause
    ClinicalMedicalReview --> BaselineRegulated: Medical Relief Provided

    BehavioralDyad --> EpisodicMatching: Query 128-dim Historical Memory
    EpisodicMatching --> CaregiverInsight: Synthesize L1-L4 Evidence & Precedents
    CaregiverInsight --> CoRegulation: Partner Delivers Scaffolding & Support
    CaregiverInsight --> AACBridge: Optional Child-Directed AAC Choice Tiles
    AACBridge --> CoRegulation: Child Selects Tile or Gestures
    CoRegulation --> BaselineRegulated: Observed De-escalation (Behavioral Resolution Verified)
```

---

## 5. Critical Analysis of Prior Art & Scientific Precedents

Project N builds directly upon and synthesizes several empirical research bodies:

### 5.1 Bioacoustics of Non-Verbal Vocalizations
- **The ReCANVo Study ([Johnson et al., 2023](#ref-7); [Narain et al., 2022](#ref-12)):**
  MIT Media Lab researchers compiled a database of 7,077 real-world communicative and affective non-verbal vocalizations from 8 minimally speaking individuals, annotated in real time by close family members.
  - *Key Finding:* Non-verbal vocalizations carry statistically significant acoustic markers of valence and communicative intent, with personalized (within-subject) models substantially outperforming generalized population-level models.
  - *Engineering Implication:* Validates Project N's individualized (N-of-1) modeling strategy, proving that non-verbal vocalizations are structured physical signals, not random noise.
- **Acoustic Voice Features in Autism Meta-Analyses ([Fusaroli et al., 2017](#ref-4)):**
  Systematic reviews demonstrate that across populations, acoustic differences between autistic and neurotypical individuals show modest effect sizes ($d = 0.4–0.5$) with only $\sim 61–64\%$ discriminatory accuracy.
  - *Clinical Implication:* Confirms that population-wide "autism voice classifiers" fail. Only an individualized, longitudinal N-of-1 acoustic baseline can reliably identify a specific child's subtle shifts.

### 5.2 Wearable Biosensing & Autonomic Forecasting
- **Biosensing in Minimally Verbal Autistic Youth ([Goodwin et al., 2019](#ref-5); [Imbiriba et al., 2023](#ref-6)):**
  In a foundational inpatient study of 20 youth with autism spectrum disorder (85% minimally verbal) across 1,000+ hours of continuous wearable biosensing (electrodermal activity, accelerometry), Goodwin et al. ([2019](#ref-5)) demonstrated that imminent aggressive distress episodes could be predicted 1 minute in advance with an AUROC of **0.84** using person-dependent time-series models (versus 0.71 for population models). 
  Subsequently, in a larger cohort of 70 psychiatric inpatients across 497 observation hours, Imbiriba et al. ([2023](#ref-6)) expanded prediction horizons to 3 minutes ahead (population AUROC 0.87 vs. person-dependent 0.74), highlighting that effective longitudinal deployment requires continuous personal calibration to match individual autonomic baselines.
  - *Engineering Implication:* Direct autonomic sensing breaks the circularity of guessing internal arousal from surface behaviors alone, providing objective somatic telemetry to substantiate behavioral escalation.

### 5.3 Motor Stimming, Kinematics & Predictive Coding
- **Self-Stimulatory Behavior Dataset (SSBD; [Rajagopalan et al., 2013](#ref-15)):**
  Rajagopalan et al. established the SSBD benchmark for video-based detection of repetitive motor behaviors (arm flapping, head banging, spinning). Subsequent temporal ablation studies by Mondal & Washington ([2026](#ref-11)) evaluated classification performance across subsampled video rates, demonstrating that model accuracy remained resilient when subsampling at intervals of every 15 frames (~2 effective fps) for rhythmic motor behaviors, confirming that hyper-dense frame rates (60+ fps) overfit to high-frequency sensor noise without improving kinematic classification.
- **The HIPPEA Framework (High Inflexible Precision of Prediction Errors; [Van de Cruys et al., 2014](#ref-19)):**
  Reframes repetitive motor stims not as meaningless pathology, but as adaptive cognitive strategies to generate predictable sensory feedback in an overwhelming, high-prediction-error world.

### 5.4 Augmentative and Alternative Communication (AAC)
- **Speech Production Outcomes in Aided AAC ([Millar et al., 2006](#ref-10)):**
  A systematic review of 23 empirical studies across 67 individuals demonstrated that introducing aided AAC resulted in increased speech production for **89% of participants**, with 11% showing no change, and **0% exhibiting any speech decrease**. A subsequent meta-analysis by Schlosser & Wendt (2008) corroborated these findings, firmly disproving the clinical myth that AAC inhibits natural vocal development.
- **Naturalistic Developmental Behavioral Interventions (NDBI; [Schreibman et al., 2015](#ref-17); [Bruinsma et al., 2020](#ref-3)):**
  Consensus clinical guidelines demonstrate that communication development accelerates when embedded within shared, child-led everyday routines with contingent partner responsiveness.
  - *Clinical & Epistemic Implication:* Communication development is transactional and partner-supported. Where accessible, AAC serves as an empowering bridge for child-directed authorship; simultaneously, partner scaffolding and timely co-regulatory interventions—measured through prospective behavioral resolution—form the primary dyadic engine for communicative connection and de-escalation.

### 5.5 Somatic Distress & Pain Evaluation
- **Non-Communicating Children’s Pain Checklist – Postoperative Version (NCCPC-PV; [Breau et al., 2002](#ref-2)):**
  A validated 27-item clinical instrument with high internal consistency ($\alpha = 0.91$) designed specifically for caregivers and clinicians to detect physical pain in children with severe communication impairments across 6 observable subscales (Vocal, Social, Facial, Activity, Body & Limbs, Physiological; total score 0–81 over a 10-minute observation window). The ROC-derived cut-off of $\ge 6$ indicates mild pain (sensitivity 0.88, specificity 0.81), while scores $\ge 11$ indicate moderate-to-severe pain. Project N explicitly adopts $\ge 6$ as its mandatory medical escalation threshold: clinical cost asymmetry dictates that a false alarm (prompting a physical pain check when distress is communicative) carries minimal risk, whereas misclassifying acute somatic pain as communicative distress risks severe patient harm.

---

## 6. The Four-Layer Evidence Model (L1–L4)

To prevent generative AI from fabricating clinical or emotional narratives, Project N enforces a strict four-layer evidence hierarchy:

```mermaid
graph LR
    subgraph L1 [Layer 1: Measured Observation]
        L1_data["Objective Sensor Data<br/>• F0 mean 268 Hz, low variance<br/>• 3.8 Hz wrist oscillation<br/>• Tonic SCL elevation"]
    end

    subgraph L2 [Layer 2: Historical Precedents]
        L2_data["Child N Episodic Memory<br/>• 3 matching prior episodes<br/>• 2 resolved via deep pressure<br/>• 0 matched hunger"]
    end

    subgraph L3 [Layer 3: Caregiver Context]
        L3_data["Situational Antecedents<br/>• Post-school transition<br/>• 45 min since last water<br/>• Loud ambient environment"]
    end

    subgraph L4 [Layer 4: Cited Evidence]
        L4_data["Curated Literature<br/>• Van de Cruys et al. (2014)<br/>• HIPPEA uncertainty reduction<br/>• Scope: Theoretical predictive coding framework;<br/>no empirical clinical sample"]
    end

    subgraph OutputCard [Schema-Constrained Caregiver Card]
        L1_data & L2_data & L3_data & L4_data --> Card["Formatted Caregiver Card<br/>(Visual boundaries, zero added claims)"]
    end

    style L1 fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style L2 fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style L3 fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style L4 fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
```

### 6.1 The Dual-Perspective Presentation Interface (Parent View vs. Therapist View)

To ensure clinical and practical utility across different stakeholders, the structured L1–L4 evidence is rendered through a toggleable interface:

1. **Parent View (Default — Warm Co-Regulatory Scaffolding):**
   Translates dense bioacoustic physics and kinematic coordinates into accessible, compassionate, and non-pathologizing everyday language (*e.g., explaining that a high vocal hum paired with hand flapping reflects sensory overload or excitement rather than defiance or anger*). It presents concrete, low-risk co-regulatory ideas grounded in past successes (*"Offer the red squishy toy that comforted him last Tuesday", "Dim lights and provide 3 minutes of quiet space"*) with an explicit non-diagnostic notice reminding families that suggestions are gentle hypotheses to investigate, not medical directives.
2. **Therapist View (Clinical & Bioacoustic Telemetry):**
   Surfaces raw fundamental frequency ($F_0$), Cepstral Peak Prominence (CPP), CQT harmonic overtone spacing, pose oscillation frequencies (Hz), SCERTS mutual regulation categories, and exact peer-reviewed literature citations, enabling Speech-Language Pathologists and Occupational Therapists to review objective empirical progress during therapy sessions.

Both views are derived from the exact same deterministic underlying records, ensuring that plain-language translation never compromises empirical grounding.

---

## 7. Implications for Clinical Practice (SLP & OT)

### 7.1 For Speech-Language Pathologists (SLPs)
1. **Ecologically Valid Longitudinal Repertoire:** Rather than relying on 45-minute weekly clinic sessions, SLPs gain access to a continuous, objective timeline of naturalistic vocalizations and communicative bids across home and community settings.
2. **Personalized AAC Target Identification:** Identifies which situations exhibit high child-initiated communication, enabling targeted design of AAC vocabulary boards that match the child's actual lived experiences.
3. **Tracking Dyadic Interaction & Scaffolding:** Allows therapists to review how different communication partner styles (wait time, simplified linguistic modeling, visual cues) correlate with child engagement and vocal variety.

### 7.2 For Occupational Therapists (OTs)
1. **Objective Sensory Profile Tracking:** Maps repetitive motor behaviors and acoustic tension against environmental antecedents (auditory noise, room transitions, sensory overload), grounding Ayres Sensory Integration in empirical physical telemetry.
2. **Co-Regulation Efficacy Assessment:** Provides measurable data on whether specific proprioceptive or vestibular interventions (e.g., deep pressure, swinging) effectively down-regulate autonomic hyper-arousal.

---

## 8. Conclusion & Collaborative Invitation

Project N redefines the role of artificial intelligence in neurodivergent communication: **shifting from an extractive intent translator to a dyadic, transactional, and partner-empowering behavioral insight assistant with optional child-authored AAC communication.** 

By uniting speech-language pathology principles, occupational therapy sensory frameworks, high-resolution acoustic and kinematic physics, and rigorous privacy engineering on local Apple Silicon hardware, Project N offers a path toward understanding non-verbal communication that honors the child's autonomy, empowers families, and respects scientific integrity.

We warmly invite speech-language pathologists, occupational therapists, assistive technology developers, and neurodiversity researchers to review, critique, and contribute to this open-source architecture.

---

## 9. References & Academic Bibliography

1. <a id="ref-1"></a>**Barrett, L. F., Adolphs, R., Marsella, S., Martinez, A. M., & Pollak, S. D. (2019).** Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements. *Psychological Science in the Public Interest*, 20(1), 1–68. [doi:10.1177/1529100619832930](https://doi.org/10.1177/1529100619832930)
2. <a id="ref-2"></a>**Breau, L. M., Finley, G. A., McGrath, P. J., & Camfield, C. S. (2002).** Validation of the Non-communicating Children’s Pain Checklist–Postoperative Version. *Anesthesiology*, 96(3), 528–535. [doi:10.1097/00000542-200203000-00007](https://doi.org/10.1097/00000542-200203000-00007)
3. <a id="ref-3"></a>**Bruinsma, Y., Minjarez, M. B., Schreibman, L., & Stahmer, A. C. (2020).** *Naturalistic developmental behavioral interventions for autism spectrum disorder*. Paul H. Brookes Publishing. [Brookes Publishing](https://products.brookespublishing.com/Naturalistic-Developmental-Behavioral-Interventions-for-Autism-Spectrum-Disorder-P1180.aspx)
4. <a id="ref-4"></a>**Fusaroli, R., Lambrechts, A., Bang, D., Bowler, D. M., & Gaigg, S. B. (2017).** Is voice a marker for Autism spectrum disorder? A systematic review and meta-analysis. *Autism Research*, 10(3), 384–407. [doi:10.1002/aur.1678](https://doi.org/10.1002/aur.1678)
5. <a id="ref-5"></a>**Goodwin, M. S., Mazefsky, C. A., Ioannidis, S., Erdogmus, D., & Siegel, M. (2019).** Predicting aggression to others in youth with autism spectrum disorder using biosensors: A mobile clinical laboratory study. *Autism Research*, 12(8), 1286–1295. [doi:10.1002/aur.2151](https://doi.org/10.1002/aur.2151)
6. <a id="ref-6"></a>**Imbiriba, T., Demirkaya, A., Singh, P., Erdogmus, D., & Goodwin, M. S. (2023).** Biosensing to predict imminent aggression in psychiatric inpatients with autism. *JAMA Network Open*, 6(12), e2348898. [doi:10.1001/jamanetworkopen.2023.48898](https://doi.org/10.1001/jamanetworkopen.2023.48898)
7. <a id="ref-7"></a>**Johnson, K. T., Narain, J., Quatieri, T., Maes, P., & Picard, R. (2023).** ReCANVo: A database of real-world communicative and affective nonverbal vocalizations. *Scientific Data*, 10(1), 523. [doi:10.1038/s41597-023-02405-7](https://doi.org/10.1038/s41597-023-02405-7)
8. <a id="ref-8"></a>**Light, J., & McNaughton, D. (2014).** Communicative competence for individuals who require augmentative and alternative communication: A new definition for a new era of communication? *Augmentative and Alternative Communication*, 30(1), 1–18. [doi:10.3109/07434618.2014.885080](https://doi.org/10.3109/07434618.2014.885080)
9. <a id="ref-9"></a>**McLean, J. E., & Snyder-McLean, L. K. (1978).** *A transactional approach to early language training*. Charles E. Merrill Publishing. [ERIC: ED172561](https://eric.ed.gov/?id=ED172561)
10. <a id="ref-10"></a>**Millar, D. C., Light, J. C., & Schlosser, R. W. (2006).** The impact of augmentative and alternative communication intervention on speech production of individuals with autism: A research review. *Journal of Speech, Language, and Hearing Research*, 49(2), 248–264. [doi:10.1044/1092-4388(2006/021)](https://doi.org/10.1044/1092-4388(2006/021))
11. <a id="ref-11"></a>**Mondal, A., & Washington, P. (2026).** Frame rate ablation in video-based autism behavior detection. *arXiv preprint arXiv:2607.07957*. [doi:10.48550/arXiv.2607.07957](https://doi.org/10.48550/arXiv.2607.07957)
12. <a id="ref-12"></a>**Narain, J., Johnson, K. T., Quatieri, T., Picard, R., & Maes, P. (2022).** Modeling real-world affective and communicative nonverbal vocalizations from minimally speaking individuals. *IEEE Transactions on Affective Computing*, 14(4), 3122–3135. [doi:10.1109/TAFFC.2022.3208233](https://doi.org/10.1109/TAFFC.2022.3208233)
13. <a id="ref-13"></a>**National Autism Center. (2026).** *Position statement on Facilitated Communication and Rapid Prompting Method*. Published June 23, 2026. [nationalautismcenter.org](https://nationalautismcenter.org/position-statements/)
14. <a id="ref-14"></a>**Prizant, B. M., Wetherby, A. M., Rubin, E., & Laurent, A. C. (2006).** *The SCERTS model: A comprehensive educational approach for children with autism spectrum disorders*. Paul H. Brookes Publishing. [scerts.com](https://scerts.com/)
15. <a id="ref-15"></a>**Rajagopalan, S. S., Dhall, A., & Goecke, R. (2013).** Self-stimulatory behaviours in the wild for autism diagnosis. *IEEE International Conference on Computer Vision Workshops (ICCVW)*, 755–761. [doi:10.1109/ICCVW.2013.103](https://doi.org/10.1109/ICCVW.2013.103)
16. <a id="ref-16"></a>**Sameroff, A. J. (1975).** Transactional models in early social relations. *Human Development*, 18(1-2), 65–79. [doi:10.1159/000271476](https://doi.org/10.1159/000271476)
17. <a id="ref-17"></a>**Schreibman, L., Dawson, G., Stahmer, A. C., Landa, R., Rogers, S. J., McGee, G. G., & Halladay, A. (2015).** Naturalistic developmental behavioral interventions: Empirically validated treatments for autism spectrum disorder. *Journal of Autism and Developmental Disorders*, 45(8), 2411–2428. [doi:10.1007/s10803-015-2407-8](https://doi.org/10.1007/s10803-015-2407-8)
18. <a id="ref-18"></a>**Tager-Flusberg, H., & Kasari, C. (2013).** Minimally verbal school-aged children with autism spectrum disorder: The neglected end of the spectrum. *Autism Research*, 6(6), 468–478. [doi:10.1002/aur.1329](https://doi.org/10.1002/aur.1329)
19. <a id="ref-19"></a>**Van de Cruys, S., Evers, K., Van der Hallen, R., Van Eylen, L., Boets, B., de-Wit, L., & Wagemans, J. (2014).** Precise minds in uncertain worlds: Predictive coding in autism. *Psychological Review*, 121(4), 649–675. [doi:10.1037/a0037665](https://doi.org/10.1037/a0037665)
20. <a id="ref-20"></a>**Wetherby, A. M., & Prizant, B. M. (2000).** *Autism spectrum disorders: A transactional developmental perspective*. Paul H. Brookes Publishing. [Brookes Publishing](https://products.brookespublishing.com/Autism-Spectrum-Disorders-P198.aspx)
