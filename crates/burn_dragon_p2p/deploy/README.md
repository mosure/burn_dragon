# burn_dragon_p2p Deploy

This folder contains the operator-facing deployment assets for the `burn_dragon_p2p` network:

- `native-peer.toml.example`: example native peer config
- `burn-dragon-p2p-native.service`: example native peer systemd unit
- `profiles/`: checked-in Dragon experiment profile sources and initial published profile payloads
- `datasets/`: checked-in browser-facing static shard datasets served from the edge
- `terraform/aws`: the checked-in AWS bootstrap/edge deployment

The AWS Terraform root deploys a single-region bootstrap plane for the Dragon network:

- one EC2 bootstrap/edge host
- Route53 DNS for the browser edge
- TLS termination via Caddy on the host
- single-host encrypted EBS-backed local publication/state storage
- configurable browser/native auth flow through `burn-p2p-bootstrap`

It does not deploy end-user native trainer peers. Native operators still install and run `burn_dragon_p2p_native` locally, then point it at the deployed edge and seed URLs.

The bootstrap publishes initial Dragon experiment directory entries for:

- `nca-prepretraining`
- `climbmix-pretraining`

Those entries include Dragon profile metadata, so peers can resolve experiment and training configuration from the network instead of requiring a matching static local config.
The initial ClimbMix revision also includes a browser shard-manifest source at
`/dragon-datasets/climbmix-pretraining/climbmix-r1/fetch-manifest.json`, served directly by the
bootstrap edge from the checked-in dataset bundle in this folder. That checked-in dataset is only a
bootstrap-compatible starter slice. For a real deployment, point the initial ClimbMix browser
profile at a full external shard pool so browser peers can fetch only the shards they train on.

## One-Click GitHub Action

The intended operator entrypoint is:

- `.github/workflows/deploy-burn-dragon-p2p-aws.yml`

That workflow:

- seeds auth client credentials into AWS SSM Parameter Store when the selected auth connector needs them
- runs `terraform fmt`, `init`, `validate`, `plan`, and `apply`
- configures explicit GitHub admin logins for session-authenticated admin access when the auth connector is `github`
- waits for the edge URL to answer over HTTPS
- prints the edge URL and seed multiaddrs in the workflow summary

If you trigger the workflow with a forced bootstrap replacement, the EC2 host is replaced and the local bootstrap/publication state on that host is lost. Use that option only for explicit rebuilds.

The workflow still performs a Terraform plan internally before apply. That keeps the operator experience one-click without dropping the safety and auditability of a plan phase.

## Required GitHub Environment Configuration

Create one or more GitHub Environments. Recommended names:

- `burn-dragon-p2p-staging`
- `burn-dragon-p2p-production`

Configure the workflow to target one of those environments. Put the following values on the selected environment.

### Required Environment Variables

- `BURN_DRAGON_P2P_AWS_ROLE_ARN`
  - AWS IAM role assumed through GitHub OIDC.
- `BURN_DRAGON_P2P_AWS_REGION`
  - AWS region for the stack, for example `us-east-1`.
- `BURN_DRAGON_P2P_STACK_NAME`
  - Terraform stack prefix, for example `burn-dragon-p2p-mainnet`.
- `BURN_DRAGON_P2P_EDGE_DOMAIN_NAME`
  - Public browser edge hostname, for example `dragon-net.example.com`.
- `BURN_DRAGON_P2P_ROUTE53_ZONE_NAME`
  - Route53 public zone name containing the edge record, for example `example.com`.
- `BURN_DRAGON_P2P_NETWORK_ID`
  - burn_p2p network id, for example `burn-dragon-mainnet`.
- `BURN_DRAGON_P2P_PROJECT_FAMILY_ID`
  - burn_p2p project family id, usually `burn-dragon-language`.
- `BURN_DRAGON_P2P_STUDY_ID`
  - study id advertised in the experiment directory.
- `BURN_DRAGON_P2P_RELEASE_TRAIN_HASH`
  - release-train hash enforced by the auth portal.
- `BURN_DRAGON_P2P_BOOTSTRAP_GIT_REF`
  - pinned `burn_p2p` git ref used to install `burn_p2p_bootstrap` on the edge host.

### Optional Environment Variables

- `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND`
  - auth connector kind for the bootstrap edge. Supported values:
    - `github`
    - `oidc`
    - `oauth`
    - `static`
    - `external`
  - defaults to `github`
- `BURN_DRAGON_P2P_AUTH_AUTHORITY_NAME`
  - logical authority label for the auth portal. Defaults to `burn-dragon-auth`.
- `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`
  - optional JSON array of seeded auth principals. This is the generic way to inject admin/operator principals for non-GitHub deployments, and it also works alongside GitHub auth if you want extra static principals.
- `BURN_DRAGON_P2P_AUTH_AUTHORIZE_BASE_URL`
- `BURN_DRAGON_P2P_AUTH_EXCHANGE_URL`
- `BURN_DRAGON_P2P_AUTH_TOKEN_URL`
- `BURN_DRAGON_P2P_AUTH_API_BASE_URL`
- `BURN_DRAGON_P2P_AUTH_USERINFO_URL`
- `BURN_DRAGON_P2P_AUTH_REFRESH_URL`
- `BURN_DRAGON_P2P_AUTH_REVOKE_URL`
- `BURN_DRAGON_P2P_AUTH_JWKS_URL`
  - optional connector endpoint overrides
- `BURN_DRAGON_P2P_AUTH_OIDC_ISSUER`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=oidc`
- `BURN_DRAGON_P2P_AUTH_OAUTH_PROVIDER`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=oauth`
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_AUTHORITY`
  - required when `BURN_DRAGON_P2P_AUTH_CONNECTOR_KIND=external`
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_TRUSTED_PRINCIPAL_HEADER`
  - trusted upstream header carrying the authenticated principal for `external` auth. Defaults to `x-forwarded-user`.
- `BURN_DRAGON_P2P_AUTH_EXTERNAL_TRUSTED_INTERNAL_ONLY`
  - whether the external connector should trust only internal ingress traffic. Defaults to `true`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_ORG`
  - GitHub org required for peer admission when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_REPO`
  - repo permission gate used for peer admission when `auth_connector_kind=github`, for example `mosure/burn_bdh`.
- `BURN_DRAGON_P2P_GITHUB_REQUIRED_TEAM`
  - optional `org/team` rule for GitHub peer admission
- `BURN_DRAGON_P2P_GITHUB_ADMIN_LOGINS`
  - optional comma-separated fallback GitHub username handles for the deploy workflow. These are plain GitHub login handles like `mosure`, not display names or email addresses. The workflow lowercases and deduplicates them. If the manual workflow input is empty, the workflow uses this value; if both are empty, it defaults to the triggering GitHub account. This setting only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_GITHUB_ADMIN_REQUIRED_REPO_PERMISSION`
  - optional minimum GitHub repository permission for explicitly listed admin logins. Defaults to `admin`. This only applies when `auth_connector_kind=github`.
- `BURN_DRAGON_P2P_INSTANCE_TYPE`
  - override bootstrap host size, default `t3.large`
- `BURN_DRAGON_P2P_ROOT_VOLUME_SIZE_GIB`
  - override encrypted EBS root size, default `256`
- `BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL`
  - optional public base URL for the full browser ClimbMix shard pool. When set, the deploy
    workflow publishes `${base_url}/fetch-manifest.json` into the initial ClimbMix browser profile
    instead of the checked-in bootstrap dataset bundle.

### Required Environment Secrets

- `BURN_DRAGON_P2P_AUTH_CLIENT_ID`
- `BURN_DRAGON_P2P_AUTH_CLIENT_SECRET`
  - generic OAuth/OIDC client credentials used when the selected auth connector needs them
- `BURN_DRAGON_P2P_GITHUB_CLIENT_ID`
- `BURN_DRAGON_P2P_GITHUB_CLIENT_SECRET`
  - legacy GitHub-specific secret names still accepted as a fallback when `auth_connector_kind=github`

The workflow writes these secrets into AWS SSM Parameter Store before `terraform apply` only when the selected auth connector needs client credentials, so they do not need to be committed into Terraform files or `.tfvars`.

There is intentionally no shared bootstrap admin token in the production flow. Admin actions are authenticated with a short-lived session id. For GitHub auth, admin capability is granted only to explicitly listed GitHub username handles that also satisfy the org/team/repo policy. For non-GitHub auth, seed explicit admin principals through `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`.

## Required AWS IAM Permissions

The GitHub OIDC role must be able to:

- manage the Terraform target resources in the selected AWS account
- write and overwrite SSM parameters under the chosen secret prefix
- read Route53 hosted zone metadata

The deployed EC2 instance role is created by Terraform and only needs to read the SSM parameters that hold:

- GitHub OAuth client id
- GitHub OAuth client secret

## Dynamic Admin Flow

The deployed network supports live experiment-directory edits without redeploying the bootstrap host.

### Security model

- admin access is session-based, not token-based
- for `github` auth:
  - the deploy workflow accepts explicit GitHub admin logins
  - each login is an exact GitHub username handle, for example `mosure`
  - matching admin sessions must also satisfy the configured org/team/repo policy
  - matching admin sessions receive:
    - `operator_role = admin`
    - `admin_capabilities = all`
    - `ExperimentScope::Admin { study_id = ... }`
- for non-GitHub auth:
  - seed admin/operator principals through `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON`
  - include the needed roles/scopes directly in those principal records
- the bootstrap itself still exposes only a bounded admin action set:
  - control and lifecycle
  - auth-policy rollout
  - diagnostics, receipts, reducer-load, and trust-bundle exports
  - operator-retention prune

### Example seeded principals for non-GitHub auth

Use `BURN_DRAGON_P2P_AUTH_PRINCIPALS_JSON` to inject principals like:

```json
[
  {
    "principal_id": "burn-dragon-admin",
    "display_name": "burn_dragon admin",
    "org_memberships": [],
    "group_memberships": [],
    "granted_roles": { "roles": ["TrainerGpu", "Validator", "Archive"] },
    "granted_scopes": [
      "Connect",
      "Discover",
      { "Train": { "experiment_id": "nca-prepretraining" } },
      { "Validate": { "experiment_id": "nca-prepretraining" } },
      { "Archive": { "experiment_id": "nca-prepretraining" } },
      { "Train": { "experiment_id": "climbmix-pretraining" } },
      { "Validate": { "experiment_id": "climbmix-pretraining" } },
      { "Archive": { "experiment_id": "climbmix-pretraining" } },
      { "Admin": { "study_id": "burn-dragon-mainnet" } }
    ],
    "allowed_networks": ["burn-dragon-mainnet"],
    "custom_claims": {
      "operator_role": "admin",
      "admin_capabilities": "all"
    }
  }
]
```

### Initial published profiles

The initial directory entries are seeded from:

- `crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json`
- `crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json`

The initial browser ClimbMix shard bundle is seeded from:

- `crates/burn_dragon_p2p/deploy/datasets/dragon-datasets/climbmix-pretraining/climbmix-r1/`

If `BURN_DRAGON_P2P_CLIMBMIX_BROWSER_DATASET_BASE_URL` is set in the selected GitHub environment
or passed as a workflow input, Terraform overrides the initial ClimbMix browser profile so it
points at the external shard pool instead of this checked-in bundle. Browser peers still fetch
only the shards they train on. With a runtime-provided training lease they use the exact assigned
microshards; otherwise they use the bounded deterministic per-peer fallback advertised by the
profile. The shipped Dragon browser app now reads that persisted browser training lease
automatically before local training starts.

Those profile payloads are derived from the source configs in the same folder. To regenerate a profile locally:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/nca-r1.training.toml \
  --experiment-kind nca \
  --output crates/burn_dragon_p2p/deploy/profiles/nca-r1.profile.json
```

For ClimbMix, pass the revision id so the browser shard-manifest URL is included in the profile:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.training.toml \
  --experiment-kind climbmix \
  --revision-id climbmix-r1 \
  --output crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json
```

To point the published ClimbMix browser profile at a full shard pool:

```bash
cargo run -p burn_dragon_p2p --features native --bin burn_dragon_p2p_native -- \
  build-profile \
  --training-config crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.training.toml \
  --experiment-kind climbmix \
  --revision-id climbmix-r1 \
  --browser-climbmix-manifest-url https://datasets.example.com/climbmix/fetch-manifest.json \
  --output crates/burn_dragon_p2p/deploy/profiles/climbmix-r1.profile.json
```

### Roll out an updated experiment profile

1. Prepare or edit the Dragon training config locally.
2. Build the new profile JSON or keep the config path for direct rollout.
3. Authenticate as an admin user.
4. Roll the updated directory entry through the bootstrap admin API using the native operator binary.

Example:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://dragon-net.example.com

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  complete-github-login \
  --config /path/to/native-peer.toml \
  --pending /path/to/pending-login.json \
  --provider-code '<github-code>' \
  --auth-bundle-out /path/to/admin-auth.json

cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  admin-rollout-profile \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --auth-bundle /path/to/admin-auth.json
```

`admin-rollout-profile` uses the current local Dragon config and manifest metadata as the source of truth and pushes a replacement directory entry through `RolloutAuthPolicy`. Peers that rely on network-published Dragon profiles will pick up the new config from the directory instead of requiring a matching static local training config.

## Native Peer Join After Deploy

After the workflow finishes, use the outputs from the workflow summary:

- `edge_url`
- `seed_node_tcp_multiaddr`
- `seed_node_quic_multiaddr`

Then run the native operator binary against that network, for example:

```bash
cargo run -p burn_dragon_p2p --features native,wgpu --bin burn_dragon_p2p_native -- \
  begin-github-login \
  --config /path/to/native-peer.toml \
  --experiment-kind nca \
  --backend wgpu \
  --edge-url https://dragon-net.example.com \
  --seed-node-url /dns4/dragon-net.example.com/tcp/4001 \
  --seed-node-url /dns4/dragon-net.example.com/udp/4001/quic-v1
```

Native peers can leave `training_config_paths` empty and rely on the published Dragon profile metadata for experiments that have a compatible network profile.

## Browser Join After Deploy

Open the deployed `edge_url` in a browser, sign in with GitHub, and join the network from the embedded browser edge UI.

Browser peers can train directly from the published Dragon profile metadata for experiments that include a browser-capable profile.

## Terraform Root

The AWS Terraform root lives at:

- `crates/burn_dragon_p2p/deploy/terraform/aws`

Use `terraform.tfvars.example` as the starting point for any local/manual run. The GitHub Action does not require a checked-in `.tfvars` file as long as the environment variables and secrets above are configured.
