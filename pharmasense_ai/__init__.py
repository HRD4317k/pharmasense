"""PharmaSense AI - evidence-grounded mechanism reasoning on top of the FAERS statistics.

Three capabilities (see docs/ARCHITECTURE.md):
  Discover  find drug / reaction relationships that are not documented in labels or literature
  Explain   trace a relationship through structure -> target -> enzyme -> pathway with sourced evidence
  Challenge look for contradicting evidence, missing evidence and alternative explanations

Everything here uses free public APIs that an ordinary person can call (see .env.example).
Every step degrades gracefully: a source that is unreachable becomes a recorded *gap*, never a crash.
"""
__version__ = "0.1.0"
