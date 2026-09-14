//! File names of the MaleCNS v1.0 flat-connectome tables, as fetched by
//! `tools/fetch-malecns.ps1`.

pub const ANNOTATIONS: &str = "body-annotations-male-cns-v1.0-minconf-0.5.feather";
pub const TRANSMITTERS: &str = "body-neurotransmitters-male-cns-v1.0.feather";
pub const WEIGHTS: &str = "connectome-weights-male-cns-v1.0-minconf-0.5.feather";

/// Low-resolution neuroglancer precomputed skeletons, one object per body id,
/// vertices in nanometres.
pub const SKELETON_URL_PREFIX: &str = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-precomputed";

/// The pack file `fetch-skeletons` writes, next to the tables.
pub const SKELETON_PACK: &str = "skeletons-precomputed.pack";
