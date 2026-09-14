# Brain viewer

The second window flypet opens shows the brain from the front, dorsal at the
top, every neuron's branching morphology drawn as lines and lit as it fires.
The nerve cord is not drawn.

## Colour key

Colour follows the functional region a neuron belongs to, from the MaleCNS
`class` and `superclass` annotations, with a small per-type hue shift so
neighbours within a region differ.

| Colour | Region |
|---|---|
| Gold | Mushroom body Kenyon cells |
| Orange | Mushroom body output neurons |
| Red-orange | Dopaminergic neurons |
| Magenta | Central complex |
| Greens | Antennal lobe: olfactory receptor neurons, projection and local neurons |
| Lime | Gustatory receptor neurons |
| Yellow-green | Mechanosensory neurons |
| Aqua | Hygro-, thermo- and chemosensory neurons |
| Cyan | Visual projection neurons |
| Teal | Optic lobe columns (held back, they are a third of all vertices) |
| Near white | Descending neurons |
| Blue-white | Ascending neurons |
| Lavender grey | Unclassified central-brain neurons (half of all vertices) |

A firing neuron's whole tree brightens to a whitened version of its colour
and fades over about 180 ms. Cell bodies are drawn as small points on top;
`--hide-somas` turns them off.

## How light is composed

Lines add their light rather than painting over each other, so dense
tracts read as bright bundles. Three factors scale a silent line:

- **Region weight**, so the two undifferentiated masses stay muted.
- **Depth**: anterior branches are brightest, posterior ones keep a quarter,
  which gives the flat projection its structure.
- **Tip fade**: every fibre eases out to nothing over its last 24 µm
  before a tip. The reconstruction's mask cut through the lateral optic
  lobes in axis-aligned 4 µm steps, and the cut ends of long axons pile up
  in planes that would otherwise draw as walls and tiles; the fade turns
  each into a gradient. A tip is a strip end no other strip shares (shared
  ends are branch points), and an end lying exactly on a chunk plane was
  cut there by the packer, so it is not faded.

## Artifacts in the source skeletons

The published skeletons were computed in 4 µm chunks. Two consequences are
handled in `connectome-convert build-skeletons`: terminal twigs sprouting
from chunk faces are pruned along with all twigs under 20 µm, iteratively,
and any segment lying flat in a chunk plane (about 1.5% of them) is removed,
since across thousands of neurons they line up into a visible grid.
