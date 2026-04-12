variable "aws_region" {
  description = "AWS region for the burn_dragon_p2p bootstrap deployment."
  type        = string
  default     = "us-east-1"
}

variable "stack_name" {
  description = "Logical stack name used for tags and DNS outputs."
  type        = string
}

variable "environment_name" {
  description = "Human-readable environment label."
  type        = string
  default     = "production"
}

variable "route53_zone_name" {
  description = "Public Route53 hosted zone name that should contain the edge record."
  type        = string
}

variable "edge_domain_name" {
  description = "Public domain served by the burn_p2p bootstrap browser edge."
  type        = string
}

variable "bootstrap_git_repository" {
  description = "Git repository used to install burn_p2p_bootstrap on the edge host."
  type        = string
  default     = "https://github.com/aberration-technology/burn_p2p.git"
}

variable "bootstrap_git_ref" {
  description = "Pinned burn_p2p git ref used to install burn_p2p_bootstrap."
  type        = string
}

variable "secret_parameter_prefix" {
  description = "SSM parameter prefix used for runtime secrets read by the bootstrap host."
  type        = string
}

variable "network_id" {
  description = "burn_p2p network id exposed by the bootstrap edge."
  type        = string
  default     = "burn-dragon-mainnet"
}

variable "auth_connector_kind" {
  description = "Bootstrap auth connector kind. Supported values: github, oidc, oauth, static, external."
  type        = string
  default     = "github"

  validation {
    condition = contains(
      ["github", "oidc", "oauth", "static", "external"],
      lower(trimspace(var.auth_connector_kind))
    )
    error_message = "auth_connector_kind must be one of: github, oidc, oauth, static, external."
  }
}

variable "auth_authority_name" {
  description = "Logical authority name exposed by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-auth"
}

variable "auth_principals_json" {
  description = "Optional JSON array of bootstrap auth principals. Use this for static, external, oidc, or oauth deployments that need seeded operator/admin principals without GitHub provider-policy rules."
  type        = string
  default     = "[]"

  validation {
    condition     = can(jsondecode(var.auth_principals_json))
    error_message = "auth_principals_json must be valid JSON."
  }
}

variable "auth_authorize_base_url" {
  description = "Optional auth connector authorize endpoint override."
  type        = string
  default     = ""
}

variable "auth_exchange_url" {
  description = "Optional auth connector token exchange endpoint override."
  type        = string
  default     = ""
}

variable "auth_token_url" {
  description = "Optional auth connector token endpoint override."
  type        = string
  default     = ""
}

variable "auth_api_base_url" {
  description = "Optional auth connector API base URL override. Primarily useful for GitHub-compatible providers."
  type        = string
  default     = ""
}

variable "auth_userinfo_url" {
  description = "Optional auth connector userinfo endpoint override."
  type        = string
  default     = ""
}

variable "auth_refresh_url" {
  description = "Optional auth connector refresh endpoint override."
  type        = string
  default     = ""
}

variable "auth_revoke_url" {
  description = "Optional auth connector revoke endpoint override."
  type        = string
  default     = ""
}

variable "auth_jwks_url" {
  description = "Optional auth connector JWKS endpoint override."
  type        = string
  default     = ""
}

variable "auth_oidc_issuer" {
  description = "OIDC issuer URL when auth_connector_kind = oidc."
  type        = string
  default     = ""
}

variable "auth_oauth_provider" {
  description = "Provider label when auth_connector_kind = oauth."
  type        = string
  default     = ""
}

variable "auth_external_authority" {
  description = "Trusted external authority label when auth_connector_kind = external."
  type        = string
  default     = ""
}

variable "auth_external_trusted_principal_header" {
  description = "Trusted ingress header carrying the authenticated principal when auth_connector_kind = external."
  type        = string
  default     = "x-forwarded-user"
}

variable "auth_external_trusted_internal_only" {
  description = "Whether the external connector should trust only internal traffic for the principal header."
  type        = bool
  default     = true
}

variable "project_family_id" {
  description = "Project family id required by the GitHub auth portal."
  type        = string
  default     = "burn-dragon-language"
}

variable "study_id" {
  description = "Study id shared by the Dragon experiment directory entries."
  type        = string
  default     = "burn-dragon-mainnet"
}

variable "release_train_hash" {
  description = "Required release train hash enforced by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-mainnet-train"
}

variable "native_target_artifact_hash" {
  description = "Artifact hash label granted to native peers by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-native"
}

variable "browser_target_artifact_hash" {
  description = "Artifact hash label granted to browser peers by the bootstrap auth portal."
  type        = string
  default     = "burn-dragon-browser"
}

variable "github_required_org" {
  description = "GitHub organization required for peer admission."
  type        = string
  default     = ""
}

variable "github_required_team" {
  description = "Optional GitHub team slug (org/team) required for peer admission."
  type        = string
  default     = ""
}

variable "github_required_repo" {
  description = "Repository access rule used for peer admission."
  type        = string
  default     = "mosure/burn_bdh"
}

variable "github_admin_logins" {
  description = "Explicit GitHub logins granted session-authenticated admin rights for this deployment."
  type        = list(string)
  default     = []
}

variable "github_admin_required_repo_permission" {
  description = "Minimum GitHub repository permission required for explicitly listed admin logins."
  type        = string
  default     = "admin"
}

variable "climbmix_browser_dataset_base_url" {
  description = "Optional public base URL for the full browser ClimbMix shard pool. When set, the initial ClimbMix browser profile points at the base URL plus /fetch-manifest.json instead of the checked-in bootstrap dataset bundle."
  type        = string
  default     = ""
}

variable "github_principal_id" {
  description = "Principal id assigned to admitted GitHub operators."
  type        = string
  default     = "burn-dragon-contributor"
}

variable "instance_type" {
  description = "EC2 instance type for the bootstrap edge."
  type        = string
  default     = "t3.large"
}

variable "root_volume_size_gib" {
  description = "Root EBS volume size for the bootstrap edge instance."
  type        = number
  default     = 256
}

variable "ssh_cidr_blocks" {
  description = "Optional SSH ingress CIDRs. Leave empty to rely on SSM/sessionless operation."
  type        = list(string)
  default     = []
}

variable "p2p_port" {
  description = "TCP/UDP port exposed for libp2p bootstrap traffic."
  type        = number
  default     = 4001
}

variable "http_port" {
  description = "Local burn_p2p bootstrap HTTP port behind Caddy."
  type        = number
  default     = 8787
}

variable "protocol_version" {
  description = "burn_p2p protocol version announced by the bootstrap edge."
  type        = string
  default     = "0.1.0"
}

variable "remaining_work_units" {
  description = "Bootstrap-side work unit budget surfaced to the directory."
  type        = number
  default     = 1000000
}

variable "local_artifact_retention_ttl_secs" {
  description = "Retention TTL for the local publication store."
  type        = number
  default     = 604800
}

variable "local_artifact_max_size_bytes" {
  description = "Maximum artifact size accepted by the local publication store."
  type        = number
  default     = 21474836480
}

variable "nca_min_device_memory_bytes" {
  description = "Advertised minimum trainer memory requirement for the NCA workload."
  type        = number
  default     = 2147483648
}

variable "climbmix_min_device_memory_bytes" {
  description = "Advertised minimum trainer memory requirement for the ClimbMix workload."
  type        = number
  default     = 6442450944
}

variable "min_system_memory_bytes" {
  description = "Advertised minimum system memory requirement for Dragon trainers."
  type        = number
  default     = 8589934592
}

variable "tags" {
  description = "Additional AWS tags applied to all resources."
  type        = map(string)
  default     = {}
}
