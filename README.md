🔐 McEliece Post-Quantum Cryptosystem Simulator

A research-oriented simulation framework for analyzing a code-based post-quantum cryptosystem under realistic communication constraints.

This project implements a McEliece-inspired encryption scheme extended with:

Binary Symmetric Channel (BSC)

CRC-based residual error detection

Stop-and-Wait ARQ retransmission

Batch simulations & statistical analysis

Grid Sweep parameter exploration

Interactive GUI

The simulator enables experimental evaluation of the security–reliability trade-off in code-based cryptography.

🚀 Motivation

Classical public-key systems (RSA, ECC) are vulnerable to quantum attacks.
McEliece, based on the hardness of general decoding, is one of the oldest and most studied post-quantum cryptosystems.

However, most analyses focus purely on cryptographic hardness.

This project explores:

How do intentional cryptographic errors interact with real communication noise?

It bridges cryptography and communication theory in a unified simulation environment.

🧠 Core Concepts

Binary Linear Codes over GF(2)

Hamming and Golay codes

Syndrome-based decoding

McEliece-style public key construction

Intentional error injection

Binary Symmetric Channel (BSC)

CRC residual error detection

Stop-and-Wait ARQ

Statistical validation & combinatorial comparison

🗂 Project Structure
.
├── Core_channel.py      # Cryptographic & communication logic
├── GUI_channel.py       # Graphical User Interface
├── README.md
└── (optional) CSV exports

Core_channel.py

Implements:

Linear code operations

McEliece-style encryption/decryption

BSC channel simulation

CRC computation & validation

ARQ retransmission logic

Batch simulations

Grid Sweep engine

Performance metrics collection

GUI_channel.py

Provides:

Interactive simulation modes

Parameter configuration

Real-time statistics display

Graph visualization (matplotlib)

CSV export support

⚙️ Installation
Requirements

Python 3.8+

numpy

matplotlib

Install dependencies:

pip install numpy matplotlib

▶️ Running the Simulator

Launch the GUI:

python GUI_channel.py


You will see three simulation modes:

🔹 Single Run

Step-by-step encryption/decryption demonstration.

🔹 Batch Simulation

Run multiple messages and compute:

Decoder Success Rate

CRC Detection Rate

False Positive / False Negative

Average Attempts per Message

Total Transmissions

🔹 Grid Sweep

Explore system behavior over ranges of:

Channel error probability (p)

Number of intentional errors

📊 Research Capabilities

This framework allows you to:

Validate decoder behavior against combinatorial theory

Study residual error probability

Analyze mis-correction events

Quantify retransmission overhead

Explore the security vs reliability trade-off

📌 Key Insight

Reducing the number of intentional errors improves communication reliability
but decreases the combinatorial complexity of decoding attacks.

This simulator makes that trade-off measurable.

🧪 Educational Scope

This implementation:

Uses short codes (Hamming, Golay) for full analytical validation

Is designed for academic and research experimentation

Is not intended for production cryptographic deployment

📈 Possible Extensions

Larger Goppa codes

ISD complexity estimation

Soft-decision channel models

Parallelized simulation

Security-level estimation tools
