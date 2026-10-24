"""flybrain -- a lab around the eonsystemspbc/fly-brain whole-brain model.

The upstream repository is a benchmark harness: it runs two hard-coded
experiments across six simulator backends and measures wall-clock time.  This
package reuses its data and its LIF parameters but turns it into something you
can poke at: select neurons by name or brain region, drive them, silence
others, and watch which parts of the brain light up over time.

Modules
-------
data     connectome tables, region labels, sparse weight matrix, selectors
sim      LIF engine and experiment runner (PyTorch, CUDA), time-binned output
world    2D arena, and the invented wiring between it and the connectome
loop     closed loop: arena -> sensory neurons -> brain -> descending -> arena
report   self-contained HTML report: 3D brain viewer, region timelines, arena
llm      Ollama client for the local models
agent    a local model running its own optogenetics experiments
cli      command line entry point
"""

__version__ = "0.1.0"
