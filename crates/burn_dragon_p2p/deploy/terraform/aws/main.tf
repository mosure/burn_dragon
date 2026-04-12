data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_route53_zone" "selected" {
  name         = endswith(var.route53_zone_name, ".") ? var.route53_zone_name : "${var.route53_zone_name}."
  private_zone = false
}

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}

data "aws_kms_alias" "ssm" {
  name = "alias/aws/ssm"
}

locals {
  tags = merge(
    var.tags,
    {
      Application = "burn-dragon-p2p"
      Environment = var.environment_name
      ManagedBy   = "terraform"
      Stack       = var.stack_name
    },
  )

  auth_connector_kind = lower(trimspace(var.auth_connector_kind))
  auth_oauth_enabled  = contains(["github", "oidc", "oauth"], local.auth_connector_kind)
  auth_redirect_path = local.auth_connector_kind == "github" ? "/callback/github" : (
    local.auth_connector_kind == "oidc" ? "/callback/oidc" : (
      local.auth_connector_kind == "oauth" ? "/callback/oauth" : null
    )
  )
  auth_endpoint_overrides = {
    authorize_base_url = trimspace(var.auth_authorize_base_url) != "" ? trimspace(var.auth_authorize_base_url) : null
    exchange_url       = trimspace(var.auth_exchange_url) != "" ? trimspace(var.auth_exchange_url) : null
    token_url          = trimspace(var.auth_token_url) != "" ? trimspace(var.auth_token_url) : null
    api_base_url       = trimspace(var.auth_api_base_url) != "" ? trimspace(var.auth_api_base_url) : null
    userinfo_url       = trimspace(var.auth_userinfo_url) != "" ? trimspace(var.auth_userinfo_url) : null
    refresh_url        = trimspace(var.auth_refresh_url) != "" ? trimspace(var.auth_refresh_url) : null
    revoke_url         = trimspace(var.auth_revoke_url) != "" ? trimspace(var.auth_revoke_url) : null
    jwks_url           = trimspace(var.auth_jwks_url) != "" ? trimspace(var.auth_jwks_url) : null
  }
  auth_principals = try(jsondecode(var.auth_principals_json), [])
  secret_parameter_names = {
    auth_client_id     = "${var.secret_parameter_prefix}/auth_client_id"
    auth_client_secret = "${var.secret_parameter_prefix}/auth_client_secret"
  }
  auth_connector = local.auth_connector_kind == "github" ? merge(
    {
      kind = "github"
      client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
      client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
      redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
    },
    local.auth_endpoint_overrides,
  ) : (
    local.auth_connector_kind == "oidc" ? merge(
      {
        kind = "oidc"
        issuer = trimspace(var.auth_oidc_issuer)
        client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
        client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
        redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
      },
      {
        for key, value in local.auth_endpoint_overrides : key => value
        if key != "api_base_url"
      },
    ) : (
      local.auth_connector_kind == "oauth" ? merge(
        {
          kind = "oauth"
          provider = trimspace(var.auth_oauth_provider)
          client_id     = "$${BURN_P2P_AUTH_CLIENT_ID}"
          client_secret = "$${BURN_P2P_AUTH_CLIENT_SECRET}"
          redirect_uri  = "$${BURN_P2P_AUTH_REDIRECT_URI}"
        },
        {
          for key, value in local.auth_endpoint_overrides : key => value
          if key != "api_base_url"
        },
      ) : (
        local.auth_connector_kind == "external" ? {
          kind = "external"
          authority = trimspace(var.auth_external_authority)
          trusted_principal_header = trimspace(var.auth_external_trusted_principal_header)
          trusted_internal_only = var.auth_external_trusted_internal_only
        } : {
          kind = "static"
        }
      )
    )
  )

  github_required_orgs  = trimspace(var.github_required_org) == "" ? [] : [var.github_required_org]
  github_required_teams = trimspace(var.github_required_team) == "" ? [] : [var.github_required_team]
  github_admin_logins = sort(tolist(toset([
    for login in var.github_admin_logins : lower(trimspace(login))
    if trimspace(login) != ""
  ])))
  climbmix_browser_manifest_url = trimspace(var.climbmix_browser_dataset_base_url) != "" ? format(
    "%s/fetch-manifest.json",
    trimsuffix(trimspace(var.climbmix_browser_dataset_base_url), "/"),
  ) : null

  dragon_experiment_scopes = [
    { "Train" = { experiment_id = "nca-prepretraining" } },
    { "Validate" = { experiment_id = "nca-prepretraining" } },
    { "Archive" = { experiment_id = "nca-prepretraining" } },
    { "Train" = { experiment_id = "climbmix-pretraining" } },
    { "Validate" = { experiment_id = "climbmix-pretraining" } },
    { "Archive" = { experiment_id = "climbmix-pretraining" } },
  ]
  dragon_admin_scopes = concat(
    ["Connect", "Discover"],
    local.dragon_experiment_scopes,
    [{ "Admin" = { study_id = var.study_id } }],
  )
  nca_profile_json      = trimspace(file("${path.module}/../../profiles/nca-r1.profile.json"))
  climbmix_profile = jsondecode(trimspace(file("${path.module}/../../profiles/climbmix-r1.profile.json")))
  climbmix_profile_json = jsonencode(
    local.climbmix_browser_manifest_url == null ? merge(
      local.climbmix_profile,
      {
        browser = null
      }
    ) : merge(
      local.climbmix_profile,
      {
        browser = merge(
          local.climbmix_profile.browser,
          {
            train_source = merge(
              local.climbmix_profile.browser.train_source,
              {
                manifest_url = local.climbmix_browser_manifest_url
              }
            )
          }
        )
      }
    )
  )

  contributor_rule = {
    principal_id   = var.github_principal_id
    display_name   = "burn_dragon mainnet contributor"
    required_orgs  = local.github_required_orgs
    required_teams = local.github_required_teams
    required_repo_access = [
      {
        repo               = var.github_required_repo
        minimum_permission = "write"
      },
    ]
    granted_roles = {
      roles = ["TrainerGpu", "Validator", "Archive"]
    }
    granted_scopes   = concat(["Connect", "Discover"], local.dragon_experiment_scopes)
    allowed_networks = [var.network_id]
    custom_claims = {
      deployment_profile = var.environment_name
      stack              = var.stack_name
    }
  }

  admin_rules = [
    for login in local.github_admin_logins : {
      principal_id   = "github-admin-${login}"
      display_name   = "burn_dragon admin ${login}"
      provider_login = login
      required_orgs  = local.github_required_orgs
      required_teams = local.github_required_teams
      required_repo_access = [
        {
          repo               = var.github_required_repo
          minimum_permission = var.github_admin_required_repo_permission
        },
      ]
      granted_roles = {
        roles = ["TrainerGpu", "Validator", "Archive"]
      }
      granted_scopes   = local.dragon_admin_scopes
      allowed_networks = [var.network_id]
      custom_claims = {
        deployment_profile = var.environment_name
        stack              = var.stack_name
        operator_role      = "admin"
        admin_capabilities = "all"
      }
    }
  ]
  auth_provider_policy = local.auth_connector_kind == "github" ? {
    github = {
      rules = concat([local.contributor_rule], local.admin_rules)
    }
  } : null

  experiment_directory = [
    {
      network_id        = var.network_id
      study_id          = var.study_id
      experiment_id     = "nca-prepretraining"
      workload_id       = "nca-prepretraining"
      display_name      = "NCA pre-pre-training"
      model_schema_hash = "burn-dragon-language-nca-v1"
      dataset_view_id   = "burn-dragon-universality-nca-v1"
      resource_requirements = {
        minimum_roles               = ["TrainerGpu"]
        minimum_device_memory_bytes = var.nca_min_device_memory_bytes
        minimum_system_memory_bytes = var.min_system_memory_bytes
        estimated_download_bytes    = 134217728
        estimated_window_seconds    = 60
      }
      visibility          = "OptIn"
      opt_in_policy       = "Scoped"
      current_revision_id = "nca-r1"
      current_head_id     = null
      allowed_roles = {
        roles = ["TrainerGpu", "Validator", "Archive"]
      }
      allowed_scopes = [
        { "Train" = { experiment_id = "nca-prepretraining" } },
        { "Validate" = { experiment_id = "nca-prepretraining" } },
        { "Archive" = { experiment_id = "nca-prepretraining" } },
      ]
      metadata = {
        experiment_kind        = "nca-prepretraining"
        stack                  = var.stack_name
        dragon_profile_version = "1"
        dragon_profile_json    = local.nca_profile_json
      }
    },
    {
      network_id        = var.network_id
      study_id          = var.study_id
      experiment_id     = "climbmix-pretraining"
      workload_id       = "climbmix-pretraining"
      display_name      = "ClimbMix pre-training"
      model_schema_hash = "burn-dragon-language-climbmix-v1"
      dataset_view_id   = "burn-dragon-climbmix-v1"
      resource_requirements = {
        minimum_roles               = ["TrainerGpu"]
        minimum_device_memory_bytes = var.climbmix_min_device_memory_bytes
        minimum_system_memory_bytes = var.min_system_memory_bytes
        estimated_download_bytes    = 2147483648
        estimated_window_seconds    = 180
      }
      visibility          = "OptIn"
      opt_in_policy       = "Scoped"
      current_revision_id = "climbmix-r1"
      current_head_id     = null
      allowed_roles = {
        roles = ["TrainerGpu", "Validator", "Archive"]
      }
      allowed_scopes = [
        { "Train" = { experiment_id = "climbmix-pretraining" } },
        { "Validate" = { experiment_id = "climbmix-pretraining" } },
        { "Archive" = { experiment_id = "climbmix-pretraining" } },
      ]
      metadata = {
        experiment_kind        = "climbmix-pretraining"
        stack                  = var.stack_name
        dragon_profile_version = "1"
        dragon_profile_json    = local.climbmix_profile_json
      }
    },
  ]

  bootstrap_daemon_config = {
    spec = {
      preset = "BootstrapOnly"
      genesis = {
        network_id       = var.network_id
        protocol_version = var.protocol_version
        display_name     = "burn_dragon P2P ${title(var.environment_name)}"
        created_at       = "2026-01-01T00:00:00Z"
        metadata = {
          purpose = "burn-dragon-p2p"
          stack   = var.stack_name
        }
      }
      platform            = "Native"
      bootstrap_addresses = []
      listen_addresses = [
        "/ip4/0.0.0.0/tcp/${var.p2p_port}",
        "/ip4/0.0.0.0/udp/${var.p2p_port}/quic-v1",
      ]
      authority = null
      archive = {
        pinned_heads                 = []
        pinned_artifacts             = []
        retain_contribution_receipts = true
      }
      admin_api = {
        supported_actions = [
          "Control",
          "ExportDiagnostics",
          "ExportDiagnosticsBundle",
          "ExportHeads",
          "ExportReceipts",
          "ExportReducerLoad",
          "ExportTrustBundle",
          "OperatorRetentionPrune",
          "RolloutAuthPolicy",
        ]
        diagnostics_enabled     = true
        receipt_exports_enabled = true
      }
    }
    http_bind_addr        = "127.0.0.1:${var.http_port}"
    allow_dev_admin_token = false
    optional_services = {
      browser_edge_enabled = true
      browser_mode         = "Trainer"
      social_mode          = "Public"
      profile_mode         = "Public"
    }
    remaining_work_units = var.remaining_work_units
    admin_signer_peer_id = "burn-dragon-bootstrap-authority"
    artifact_publication = {
      targets = [
        {
          publication_target_id   = "local-default"
          label                   = "hot-local"
          kind                    = "LocalFilesystem"
          publication_mode        = "LazyOnDemand"
          access_mode             = "Authenticated"
          allow_public_reads      = false
          supports_signed_urls    = false
          edge_proxy_required     = true
          max_artifact_size_bytes = var.local_artifact_max_size_bytes
          retention_ttl_secs      = var.local_artifact_retention_ttl_secs
          allowed_artifact_profiles = [
            "FullTrainingCheckpoint",
            "ServeCheckpoint",
            "BrowserSnapshot",
            "ManifestOnly",
          ]
          eager_alias_names         = []
          local_root                = "/var/lib/burn-p2p/publication/hot"
          bucket                    = null
          endpoint                  = null
          region                    = null
          access_key_id             = null
          secret_access_key         = null
          session_token             = null
          path_prefix               = "hot"
          multipart_threshold_bytes = null
          server_side_encryption    = null
          signed_url_ttl_secs       = null
        },
      ]
    }
    bootstrap_peer = {
      node = {
        identity = "Persistent"
        storage = {
          root = "/var/lib/burn-p2p/bootstrap-peer"
        }
        dataset         = null
        bootstrap_peers = []
        listen_addresses = [
          "/ip4/0.0.0.0/tcp/${var.p2p_port}",
          "/ip4/0.0.0.0/udp/${var.p2p_port}/quic-v1",
        ]
      }
    }
    auth = {
      authority_name = var.auth_authority_name
      connector = local.auth_connector
      authority_key_path          = "/var/lib/burn-p2p/auth/bootstrap-authority.key"
      session_state_path          = "/var/lib/burn-p2p/auth/session-state.json"
      persist_provider_tokens     = false
      issuer_key_id               = "burn-dragon-mainnet"
      project_family_id           = var.project_family_id
      required_release_train_hash = var.release_train_hash
      allowed_target_artifact_hashes = [
        var.native_target_artifact_hash,
        var.browser_target_artifact_hash,
      ]
      session_ttl_seconds      = 86400
      minimum_revocation_epoch = 1
      principals        = local.auth_principals
      provider_policy   = local.auth_provider_policy
      directory_entries = local.experiment_directory
    }
  }

  bootstrap_config_json = jsonencode(local.bootstrap_daemon_config)
  caddyfile = templatefile("${path.module}/templates/Caddyfile.tftpl", {
    edge_domain_name = var.edge_domain_name
    http_port        = var.http_port
  })
  secret_sync_script = templatefile("${path.module}/templates/bootstrap-secret-sync.sh.tftpl", {
    aws_region                      = var.aws_region
    auth_client_credentials_required = local.auth_oauth_enabled
    auth_client_id_name             = local.secret_parameter_names.auth_client_id
    auth_client_secret_name         = local.secret_parameter_names.auth_client_secret
    auth_redirect_uri               = local.auth_redirect_path == null ? "" : "https://${var.edge_domain_name}${local.auth_redirect_path}"
    edge_domain_name                = var.edge_domain_name
  })
  bootstrap_auth_feature = local.auth_connector_kind == "github" ? "auth-github" : (
    local.auth_connector_kind == "oidc" ? "auth-oidc" : (
      local.auth_connector_kind == "oauth" ? "auth-oauth" : (
        local.auth_connector_kind == "external" ? "auth-external" : "auth-static"
      )
    )
  )
}

check "github_connector_configuration" {
  assert {
    condition = local.auth_connector_kind != "github" || (
      trimspace(var.github_required_org) != "" &&
      trimspace(var.github_required_repo) != ""
    )
    error_message = "github auth deployments require github_required_org and github_required_repo."
  }
}

check "oidc_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "oidc" || trimspace(var.auth_oidc_issuer) != ""
    error_message = "oidc auth deployments require auth_oidc_issuer."
  }
}

check "oauth_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "oauth" || trimspace(var.auth_oauth_provider) != ""
    error_message = "oauth auth deployments require auth_oauth_provider."
  }
}

check "external_connector_configuration" {
  assert {
    condition     = local.auth_connector_kind != "external" || trimspace(var.auth_external_authority) != ""
    error_message = "external auth deployments require auth_external_authority."
  }
}

resource "aws_vpc" "bootstrap" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = merge(local.tags, {
    Name = "${var.stack_name}-vpc"
  })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.bootstrap.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = merge(local.tags, {
    Name = "${var.stack_name}-public"
  })
}

resource "aws_internet_gateway" "bootstrap" {
  vpc_id = aws_vpc.bootstrap.id

  tags = merge(local.tags, {
    Name = "${var.stack_name}-igw"
  })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.bootstrap.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.bootstrap.id
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-public"
  })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "bootstrap" {
  name        = "${var.stack_name}-bootstrap"
  description = "burn_dragon_p2p bootstrap edge"
  vpc_id      = aws_vpc.bootstrap.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = var.p2p_port
    to_port     = var.p2p_port
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.ssh_cidr_blocks
    content {
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_iam_role" "bootstrap" {
  name = "${var.stack_name}-bootstrap"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      },
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy_attachment" "ssm_managed_core" {
  role       = aws_iam_role.bootstrap.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "bootstrap_secret_access" {
  name = "${var.stack_name}-secret-access"
  role = aws_iam_role.bootstrap.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
        ]
        Resource = [
          "arn:${data.aws_partition.current.partition}:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter${var.secret_parameter_prefix}/*",
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Decrypt",
        ]
        Resource = [data.aws_kms_alias.ssm.target_key_arn]
      },
    ]
  })
}

resource "aws_iam_instance_profile" "bootstrap" {
  name = "${var.stack_name}-bootstrap"
  role = aws_iam_role.bootstrap.name
}

resource "aws_instance" "bootstrap" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.bootstrap.id]
  iam_instance_profile        = aws_iam_instance_profile.bootstrap.name
  associate_public_ip_address = true

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  root_block_device {
    volume_size           = var.root_volume_size_gib
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/templates/user-data.sh.tftpl", {
    aws_region            = var.aws_region
    bootstrap_auth_feature = local.bootstrap_auth_feature
    bootstrap_git_ref     = var.bootstrap_git_ref
    bootstrap_git_repo    = var.bootstrap_git_repository
    bootstrap_config_json = local.bootstrap_config_json
    caddyfile             = local.caddyfile
    http_port             = var.http_port
    secret_sync_script    = local.secret_sync_script
  })

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_eip" "bootstrap" {
  domain   = "vpc"
  instance = aws_instance.bootstrap.id

  tags = merge(local.tags, {
    Name = "${var.stack_name}-bootstrap"
  })
}

resource "aws_route53_record" "edge" {
  zone_id = data.aws_route53_zone.selected.zone_id
  name    = var.edge_domain_name
  type    = "A"
  ttl     = 60
  records = [aws_eip.bootstrap.public_ip]
}
