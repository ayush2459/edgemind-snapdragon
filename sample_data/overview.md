# EdgeMind Overview

EdgeMind is a private AI workspace copilot that runs entirely on the laptop. Documents, notes and meeting recordings are indexed locally, so no document text is ever sent to a cloud service.

The Snapdragon X series includes a Hexagon NPU designed for sustained AI inference at low power. EdgeMind routes embedding and speech models to the NPU, which keeps the CPU free and preserves battery life during long work sessions.

A power-aware policy engine reads the battery state. When the laptop is plugged in, EdgeMind favours speed. On battery, it lowers NPU performance mode and batch size to extend runtime.

Users can ask questions across their files, summarise long documents and search meeting transcripts. Answers cite the source files so results can be verified.
