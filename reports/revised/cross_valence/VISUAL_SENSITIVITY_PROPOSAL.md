# Visual-region sensitivity proposal — NOT EXECUTED

This independent definition was fixed in code before new results. It is a proposal only: no source features are excluded and no classifier is fit with this mask. Prospective review and approval are required.

Source: Harvard-Oxford cort maxprob thr25 2mm; installed FSL version: 6.0.7.18.

Image: `/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz`

Image SHA256: `7397ffdbae7559e0f0aa6237f998dc0f6aca3f9db9009f6ffa637c0c62b686ca`

Label XML: `/usr/local/fsl/data/atlases/HarvardOxford-Cortical.xml`

XML SHA256: `b8eb5ba5e787a4d3b442f75041f533f31b278a83e467bd2ab22438c1cff36581`

Exact integer values and labels (XML zero-based indices +1 in maxprob image):

- 22: Lateral Occipital Cortex, superior division
- 23: Lateral Occipital Cortex, inferior division
- 24: Intracalcarine Cortex
- 32: Cuneal Cortex
- 36: Lingual Gyrus
- 40: Occipital Fusiform Gyrus
- 47: Supracalcarine Cortex
- 48: Occipital Pole

Proposed rule: nearest-neighbor resample the union of these labels to each frozen analysis grid, then exclude its intersection with that original analysis mask. Recenter each remaining map independently. Preserve participants, folds, contrasts and classifier settings. Do not optimize the threshold or parcel set.

Native union: 27220 voxels.

- partner_pair: 9760 of 84133 voxels excluded; 74373 retained.
- three_paradigm: 9760 of 84045 voxels excluded; 74285 retained.

Rationale: a fixed anatomical occipital definition spanning medial and lateral occipital cortex, lingual and occipital fusiform tissue. This is not a complete functional visual-network mask; temporal face-sensitive tissue and perceptual information elsewhere may remain. Conversely, included association cortex need not be exclusively visual. Persistence would not prove abstract social coding.

Inventory and exact asset fingerprints: `provenance/revised/cross_valence/local_anatomy.json`. Peak labels describe individual coordinates, not the identity of a whole distributed component. No atlas significance or regional necessity is inferred.

holdout_scored=False.
