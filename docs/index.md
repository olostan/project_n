# Project N: Multimodal Communication Support for Non-Verbal Autism

<p align="center">
  <em>An open-source, local-first intelligence architecture designed to support completely non-verbal autistic children through high-resolution acoustic physics, body-relative kinematics, transactional adult partner scaffolding, and child-authored AAC communication.</em>
</p>

---

## A Note from the Founder

<div style="border-left: 4px solid #1890ff; padding-left: 1rem; margin: 1.5rem 0; font-style: italic; background-color: rgba(24, 144, 255, 0.05); padding-top: 0.5rem; padding-bottom: 0.5rem;">
  <p>"I am <a href="https://olostan.me/" target="_blank"><strong>Valentyn Shybanov</strong></a>, a software engineer, systems architect, and the father of Nolan, a 7-year-old completely non-verbal boy with Level 3 autism. Every single day of my life is shaped by the profound love, challenge, and heartbreak of trying to understand my son. Nolan is non-verbal. Completely.</p>
  <p>My son cannot use spoken words, but he is never silent. He communicates continuously: through subtle pitch inflections in his throat, micro-tremors in his hands, bodily orientations, and rhythms of movement. Traditional foundation models and commercial AI discard these signals as 'meaningless background noise.' But to me, as his father, that 'noise' is his entire voice.</p>
  <p>I started Project N not as a commercial startup, not to make money, and not to promote a product. I started it out of a father’s deep desire to understand his child. I am pouring my twenty-plus years of engineering experience, systems architecture knowledge, and machine learning skills into building a free, open-source, local-first tool that can help parents like me and the dedicated therapists who support our children.</p>
  <p>If you are a speech-language pathologist, an occupational therapist, an autism researcher, or an engineer who believes in communication rights: I warmly invite you to review this work, critique it, and help us make it better."</p>
  <p align="right"><strong>— Valentyn Shybanov (<a href="https://olostan.me/">olostan.me</a> · <a href="https://github.com/olostan">GitHub</a>)</strong></p>
</div>

---

## Core System Architecture

Project N is engineered to run **100% offline** on local Apple Silicon hardware (tested on an Apple M5 Pro with 48GB Unified Memory), preserving complete family privacy while eliminating cloud latency.

```mermaid
sequenceDiagram
    autonumber
    actor Child as Child N (Completely Non-Verbal)
    actor Partner as Communication Partner (Parent / SLP / OT)
    participant Engine as Project N Local Assistant (Apple Silicon)
    participant AAC as Child's AAC Speech Device

    Child->>Partner: Vocal inflection + rhythmic wrist stim (Natural bid)
    Note over Partner: Partner observes & scaffolds interpersonal support:<br/>"Do you need a sensory break?"
    Partner->>Engine: Natural episode captured via mobile client
    Engine->>Engine: Computes pitch contour (F0), pose landmarks & physiological arousal
    Engine->>Engine: Matches against Child N's historical verified episodes (128-dim metric space)
    Engine->>AAC: Dispatches candidate options ([Water], [Sensory Break], [Deep Pressure])
    Child->>AAC: Directly selects [Sensory Break] icon
    AAC-->>Partner: Speaks aloud: "Sensory Break"
    Partner->>Child: Provides quiet space / dim lights (Co-regulation restored)
    Note over Child,Partner: Regulation latency & child confirmation logged
    Engine->>Engine: Stores child-confirmed resolution as ground truth
```

---

## The Four Foundational Pillars

```mermaid
graph LR
    P1["1. Bioacoustic Physics<br/>F0, CQT, Jitter, Shimmer, CPP<br/>(No Phonemic Collapse)"]
    P2["2. Dyadic & Transactional Loop<br/>SCERTS Model & Adult Partner<br/>(Scaffolding & Regulation)"]
    P3["3. Child Authorship (AAC)<br/>Options routed to speech device<br/>(Child selection is truth)"]
    P4["4. 100% Offline Local Vault<br/>Apple Silicon Metal & Two-Key<br/>(Zero Cloud Telemetry)"]

    P1 & P2 & P3 & P4 --> Core["Project N Communication Assistant"]

    style P1 fill:#e6f7ff,stroke:#1890ff,stroke-width:2px
    style P2 fill:#f6ffed,stroke:#52c41a,stroke-width:2px
    style P3 fill:#fffbe6,stroke:#faad14,stroke-width:2px
    style P4 fill:#f9f0ff,stroke:#722ed1,stroke-width:2px
```

1. **Acoustic Physics Without Words:** Level 3 non-verbal vocalizations are analyzed using raw bioacoustic physics (fundamental frequency $F_0$ pitch tracking at $\sim 1\text{ Hz}$ resolution, jitter, shimmer, harmonic overtones via CQT, and glottal strain via CPP). Sounds are matched directly to past verified episodes without forcing them into clumsy English descriptions.
2. **The Dyadic Transactional Loop:** Communication is an evolving interaction loop between the child and their communication partner (parent, therapist). Grounded in the **SCERTS Model** ([Prizant et al., 2006](WHITE_PAPER.md#ref-11)), the system analyzes what the adult said, what physical scaffolding was offered, and how the child responded.
3. **Child Authorship via the AAC Bridge:** The system never speaks *for* the child. Candidate possibilities are routed to Child N's speech-generating device or AAC choice board as pre-populated icons for him to select, confirm, or reject.
4. **Medical Safety Gate (NCCPC-R):** Acute distress is evaluated using the validated 27-item Non-Communicating Children’s Pain Checklist – Revised ([Breau et al., 2002](WHITE_PAPER.md#ref-2)), immediately triggering medical review escalations for physical pain (ear infections, dental abscesses, GI reflux).

---

## Documentation Roadmap

- 📄 **[Scientific Whitepaper](WHITE_PAPER.md):** Formal clinical paper for Speech-Language Pathologists, OTs, and autism researchers detailing the transactional paradigm, bioacoustics, and AAC authorship.
- 📋 **[Preregistered Evaluation Protocol](evaluation_protocol.md):** Single-case (N-of-1) study design with leave-one-day-out splits, mandatory baselines B1–B4, and safety metrics.
- 📐 **[Technical Specifications](specs.md):** Complete mathematical definitions, tensor shapes, REST endpoints, SSE event schemas, and executable MLX pseudo-code.
- 🧠 **[Theoretical Architecture & Design](design.md):** Detailed neurobiology, acoustic physics, pose kinematics, literature grounding, and continuous adaptation.
- 🛡️ **[System & Safety Invariants](invariants.md):** True non-negotiable invariants (100% offline, privacy vault, frozen LLM, medical rule-out) vs. tunable empirical defaults.
- 🔍 **[Multi-Reviewer Scientific Audit](REVIEW_REFINEMENTS.md):** Comprehensive finding-by-finding peer review and corrective action matrix.
- 🤖 **[Agent Directives](agents.md):** Engineering standards, MLX memory conventions, and mandatory documentation synchronization protocol.
