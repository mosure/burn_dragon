# External Dataset Shard Pools

Large browser shard pools are intentionally not stored in git.

For `burn_dragon_p2p` deployments, publish the full ClimbMix shard pool to external object
storage or a static CDN, then set `BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL` in the
deployment workflow environment. The deploy flow publishes `${base_url}/fetch-manifest.json` into
the initial ClimbMix browser profile, and browser peers download only the shards they actually
train on.

This folder is kept only for operator documentation; do not commit shard `.bin` payloads here.
