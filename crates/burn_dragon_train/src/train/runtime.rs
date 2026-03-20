use std::any::Any;
use std::env;
#[cfg(feature = "ddp")]
use std::str::FromStr;

use anyhow::{Result, anyhow};
use burn::tensor::backend::{Backend as BackendTrait, Device, DeviceId, DeviceOps};
#[cfg(feature = "ddp")]
use burn_collective::{AllReduceStrategy, CollectiveConfig};
#[cfg(feature = "ddp")]
use burn_communication::Address;
use burn_cubecl::cubecl::Runtime;

#[cfg(all(feature = "cuda", any(feature = "cli", feature = "train")))]
use burn_cuda::CudaDevice;
#[cfg(any(feature = "cli", feature = "train"))]
use burn_ndarray::NdArrayDevice;
#[cfg(any(feature = "cli", feature = "train"))]
use burn_wgpu::WgpuDevice;

use crate::{ParallelConfig, ParallelismKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemoryUsage {
    pub reserved_bytes: u64,
    pub in_use_bytes: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct ParallelEnv {
    world_size: Option<usize>,
    global_rank: Option<usize>,
    local_rank: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParallelRuntime {
    pub mode: ParallelismKind,
    pub world_size: usize,
    pub global_rank: usize,
    pub local_rank: usize,
    pub data_parallel_size: usize,
    pub local_data_parallel_size: usize,
    pub tensor_parallel_size: usize,
    pub process_group_launch: bool,
}

impl ParallelRuntime {
    pub fn is_primary(&self) -> bool {
        self.global_rank == 0
    }

    pub fn is_process_group_launch(&self) -> bool {
        self.process_group_launch
    }

    pub fn summary(&self) -> String {
        format!(
            "mode={:?} world_size={} global_rank={} local_rank={} dp_size={} local_dp_size={} tp_size={} process_group={}",
            self.mode,
            self.world_size,
            self.global_rank,
            self.local_rank,
            self.data_parallel_size,
            self.local_data_parallel_size,
            self.tensor_parallel_size,
            self.process_group_launch
        )
    }
}

pub fn resolve_parallel_runtime(config: &ParallelConfig) -> Result<ParallelRuntime> {
    resolve_parallel_runtime_with_env(config, &parallel_env_from_process())
}

fn parallel_env_from_process() -> ParallelEnv {
    ParallelEnv {
        world_size: env::var("WORLD_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
        global_rank: env::var("RANK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
        local_rank: env::var("LOCAL_RANK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok()),
    }
}

#[cfg(feature = "ddp")]
fn normalize_collective_address(raw: &str) -> String {
    if raw.contains("://") {
        raw.to_string()
    } else {
        format!("ws://{raw}")
    }
}

fn resolve_parallel_runtime_with_env(
    config: &ParallelConfig,
    env: &ParallelEnv,
) -> Result<ParallelRuntime> {
    match config.mode {
        ParallelismKind::Single => {
            if env.world_size.is_some_and(|value| value != 1) {
                return Err(anyhow!(
                    "parallel.mode=single but WORLD_SIZE={} in environment",
                    env.world_size.unwrap_or(1)
                ));
            }
            if env.global_rank.is_some_and(|value| value != 0) {
                return Err(anyhow!(
                    "parallel.mode=single but RANK={} in environment",
                    env.global_rank.unwrap_or(0)
                ));
            }
            if env.local_rank.is_some_and(|value| value != 0) {
                return Err(anyhow!(
                    "parallel.mode=single but LOCAL_RANK={} in environment",
                    env.local_rank.unwrap_or(0)
                ));
            }
            Ok(ParallelRuntime {
                mode: ParallelismKind::Single,
                world_size: 1,
                global_rank: 0,
                local_rank: 0,
                data_parallel_size: 1,
                local_data_parallel_size: 1,
                tensor_parallel_size: 1,
                process_group_launch: false,
            })
        }
        ParallelismKind::Ddp => {
            if let Some(world_size) = env.world_size.filter(|value| *value > 1) {
                if config.world_size != world_size {
                    return Err(anyhow!(
                        "parallel.mode=ddp process-group launch requires parallel.world_size to match WORLD_SIZE (got config={} env={world_size})",
                        config.world_size
                    ));
                }
                let global_rank = env.global_rank.ok_or_else(|| {
                    anyhow!("parallel.mode=ddp process-group launch requires RANK")
                })?;
                let local_rank = env.local_rank.ok_or_else(|| {
                    anyhow!("parallel.mode=ddp process-group launch requires LOCAL_RANK")
                })?;
                if global_rank >= world_size {
                    return Err(anyhow!(
                        "parallel.mode=ddp process-group launch requires RANK < WORLD_SIZE (got {global_rank} >= {world_size})"
                    ));
                }
                if config.data.size != world_size {
                    return Err(anyhow!(
                        "parallel.mode=ddp process-group launch currently requires parallel.data.size = WORLD_SIZE (got {} != {world_size})",
                        config.data.size
                    ));
                }

                return Ok(ParallelRuntime {
                    mode: ParallelismKind::Ddp,
                    world_size,
                    global_rank,
                    local_rank,
                    data_parallel_size: config.data.size.max(1),
                    local_data_parallel_size: 1,
                    tensor_parallel_size: 1,
                    process_group_launch: true,
                });
            }
            if env.global_rank.is_some_and(|value| value != 0) {
                return Err(anyhow!(
                    "parallel.mode=ddp local multi-device bridge requires RANK=0 when WORLD_SIZE is not set"
                ));
            }
            if env.local_rank.is_some_and(|value| value != 0) {
                return Err(anyhow!(
                    "parallel.mode=ddp local multi-device bridge requires LOCAL_RANK=0 when WORLD_SIZE is not set"
                ));
            }

            Ok(ParallelRuntime {
                mode: ParallelismKind::Ddp,
                world_size: config.world_size.max(1),
                global_rank: 0,
                local_rank: 0,
                data_parallel_size: config.data.size.max(1),
                local_data_parallel_size: config.data.size.max(1),
                tensor_parallel_size: 1,
                process_group_launch: false,
            })
        }
        mode => Err(anyhow!(
            "parallel.mode={mode:?} is configured, but the distributed runtime for this mode is not implemented yet"
        )),
    }
}

pub fn resolve_training_devices<B>(
    runtime: &ParallelRuntime,
    primary_device: &B::Device,
) -> Result<Vec<B::Device>>
where
    B: BackendTrait,
    B::Device: DeviceOps + 'static,
{
    match runtime.mode {
        ParallelismKind::Single => Ok(vec![primary_device.clone()]),
        ParallelismKind::Ddp => {
            if runtime.process_group_launch {
                Ok(vec![resolve_local_rank_device::<B>(
                    primary_device,
                    runtime.local_rank,
                )?])
            } else {
                resolve_local_multi_device_bridge::<B>(
                    primary_device,
                    runtime.local_data_parallel_size.max(1),
                )
            }
        }
        mode => Err(anyhow!(
            "parallel.mode={mode:?} does not have a training-device resolver yet"
        )),
    }
}

#[cfg(feature = "ddp")]
pub fn resolve_collective_config(
    runtime: &ParallelRuntime,
    config: &ParallelConfig,
) -> Result<CollectiveConfig> {
    match runtime.mode {
        ParallelismKind::Ddp => {
            let mut collective = CollectiveConfig::default()
                .with_num_devices(runtime.local_data_parallel_size.max(1));
            match (
                config.data.collective_num_nodes,
                config.data.collective_global_address.as_deref(),
                config.data.collective_node_address.as_deref(),
                config.data.collective_data_service_port,
            ) {
                (None, None, None, None) => {
                    if runtime.process_group_launch {
                        return Err(anyhow!(
                            "parallel.mode=ddp process-group launches require collective global settings in [parallel.data]"
                        ));
                    }
                }
                (Some(num_nodes), Some(global_address), Some(node_address), Some(port)) => {
                    let global_address_raw = normalize_collective_address(global_address);
                    let node_address_raw = normalize_collective_address(node_address);
                    let global_address = Address::from_str(&global_address_raw).map_err(|err| {
                        anyhow!(
                            "invalid parallel.data.collective_global_address `{global_address}`: {err}"
                        )
                    })?;
                    let node_address = Address::from_str(&node_address_raw).map_err(|err| {
                        anyhow!(
                            "invalid parallel.data.collective_node_address `{node_address}`: {err}"
                        )
                    })?;
                    collective = collective
                        .with_num_nodes(num_nodes)
                        .with_global_address(global_address)
                        .with_node_address(node_address)
                        .with_global_all_reduce_strategy(AllReduceStrategy::Centralized)
                        .with_data_service_port(port);
                }
                _ => {
                    return Err(anyhow!(
                        "parallel.data collective global settings must either all be set or all be omitted"
                    ));
                }
            }

            if !collective.is_valid() {
                return Err(anyhow!(
                    "resolved collective config is invalid; check parallel.data collective global settings"
                ));
            }

            Ok(collective)
        }
        mode => Err(anyhow!(
            "parallel.mode={mode:?} does not use a collective config"
        )),
    }
}

fn resolve_local_multi_device_bridge<B>(
    primary_device: &B::Device,
    replica_count: usize,
) -> Result<Vec<B::Device>>
where
    B: BackendTrait,
    B::Device: DeviceOps + 'static,
{
    if replica_count <= 1 {
        return Ok(vec![primary_device.clone()]);
    }

    #[cfg(any(feature = "cli", feature = "train"))]
    if (primary_device as &dyn Any)
        .downcast_ref::<NdArrayDevice>()
        .is_some()
    {
        return Ok((0..replica_count).map(|_| B::Device::default()).collect());
    }

    let primary_id = primary_device.id();
    let type_id = resolve_replica_type_id(primary_device, primary_id.type_id, replica_count);
    let available = <B::Device as Device>::device_count(type_id);
    if available < replica_count {
        return Err(anyhow!(
            "parallel local multi-device bridge requested {replica_count} replicas, but only {available} devices are available for type_id={type_id}"
        ));
    }

    Ok((0..replica_count)
        .map(|index| <B::Device as Device>::from_id(DeviceId::new(type_id, index as u32)))
        .collect())
}

fn resolve_local_rank_device<B>(primary_device: &B::Device, local_rank: usize) -> Result<B::Device>
where
    B: BackendTrait,
    B::Device: DeviceOps + 'static,
{
    #[cfg(any(feature = "cli", feature = "train"))]
    if (primary_device as &dyn Any)
        .downcast_ref::<NdArrayDevice>()
        .is_some()
    {
        return Ok(B::Device::default());
    }

    let primary_id = primary_device.id();
    let replica_count = local_rank.saturating_add(1);
    let type_id = resolve_replica_type_id(primary_device, primary_id.type_id, replica_count);
    let available = <B::Device as Device>::device_count(type_id);
    if available <= local_rank {
        return Err(anyhow!(
            "parallel process-group launch requested LOCAL_RANK={local_rank}, but only {available} devices are available for type_id={type_id}"
        ));
    }

    Ok(<B::Device as Device>::from_id(DeviceId::new(
        type_id,
        local_rank as u32,
    )))
}

fn resolve_replica_type_id<D: DeviceOps + 'static>(
    primary_device: &D,
    default_type_id: u16,
    replica_count: usize,
) -> u16 {
    #[cfg(any(feature = "cli", feature = "train"))]
    if replica_count > 1
        && (primary_device as &dyn Any)
            .downcast_ref::<WgpuDevice>()
            .is_some_and(|device| matches!(device, WgpuDevice::DefaultDevice))
    {
        return 0;
    }

    default_type_id
}

impl DeviceMemoryUsage {
    pub fn reserved_mb(self) -> f64 {
        bytes_to_mb(self.reserved_bytes)
    }

    pub fn in_use_mb(self) -> f64 {
        bytes_to_mb(self.in_use_bytes)
    }
}

pub fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn device_memory_usage<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        let usage =
            <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }

    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        let usage = <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }

    None
}

pub fn device_memory_usage_safe<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        device_memory_usage::<B>(device)
    }))
    .ok()
    .flatten()
}

pub fn cleanup_device_memory<B: BackendTrait>(device: &B::Device, allow_cuda_cleanup: bool) -> bool
where
    B::Device: 'static,
{
    if !cleanup_device_memory_allowed::<B>(device, allow_cuda_cleanup) {
        return false;
    }

    let _guard = crate::device::device_allocation_lock().lock().ok();
    let _ = B::sync(device);
    B::memory_cleanup(device);
    extra_memory_cleanup::<B>(device);
    let _ = B::sync(device);
    true
}

pub fn cleanup_device_memory_allowed<B: BackendTrait>(
    device: &B::Device,
    allow_cuda_cleanup: bool,
) -> bool
where
    B::Device: 'static,
{
    allow_memory_cleanup::<B>(device, allow_cuda_cleanup)
}

fn extra_memory_cleanup<B: BackendTrait>(device: &B::Device)
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_cleanup();
    }

    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_cleanup();
    }
}

fn allow_memory_cleanup<B: BackendTrait>(_device: &B::Device, _allow_cuda_cleanup: bool) -> bool
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if (_device as &dyn Any).downcast_ref::<CudaDevice>().is_some() {
        return _allow_cuda_cleanup;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::{ParallelEnv, resolve_parallel_runtime_with_env, resolve_training_devices};
    #[cfg(feature = "ddp")]
    use crate::train::runtime::resolve_collective_config;
    use crate::{ParallelConfig, ParallelismKind};
    use burn_ndarray::NdArray;
    #[cfg(feature = "ddp")]
    use serde_json::json;

    #[test]
    fn resolve_parallel_runtime_accepts_single_mode_without_env() {
        let runtime =
            resolve_parallel_runtime_with_env(&ParallelConfig::default(), &ParallelEnv::default())
                .expect("single runtime");
        assert!(runtime.is_primary());
        assert_eq!(runtime.world_size, 1);
        assert_eq!(runtime.global_rank, 0);
    }

    #[test]
    fn resolve_parallel_runtime_rejects_distributed_modes_until_runtime_exists() {
        let config = ParallelConfig {
            mode: ParallelismKind::Fsdp,
            world_size: 2,
            data: crate::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let err = resolve_parallel_runtime_with_env(&config, &ParallelEnv::default())
            .expect_err("ddp should fail before runtime support lands");
        assert!(
            err.to_string().contains("not implemented yet"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn resolve_parallel_runtime_accepts_local_ddp_bridge_without_env() {
        let config = ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: crate::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let runtime = resolve_parallel_runtime_with_env(&config, &ParallelEnv::default())
            .expect("local ddp bridge");
        assert_eq!(runtime.mode, ParallelismKind::Ddp);
        assert_eq!(runtime.world_size, 2);
        assert_eq!(runtime.data_parallel_size, 2);
        assert!(runtime.is_primary());
    }

    #[test]
    fn resolve_parallel_runtime_accepts_process_group_ddp_env() {
        let config = ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: crate::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let env = ParallelEnv {
            world_size: Some(2),
            global_rank: Some(1),
            local_rank: Some(1),
        };

        let runtime = resolve_parallel_runtime_with_env(&config, &env)
            .expect("process-group ddp runtime should resolve");
        assert_eq!(runtime.mode, ParallelismKind::Ddp);
        assert_eq!(runtime.world_size, 2);
        assert_eq!(runtime.global_rank, 1);
        assert_eq!(runtime.local_rank, 1);
        assert_eq!(runtime.data_parallel_size, 2);
        assert_eq!(runtime.local_data_parallel_size, 1);
        assert!(runtime.is_process_group_launch());
    }

    #[test]
    fn resolve_training_devices_repeats_cpu_for_ndarray_local_ddp_bridge() {
        type TestBackend = NdArray<f32>;

        let runtime = super::ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 2,
            local_data_parallel_size: 2,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };
        let devices = resolve_training_devices::<TestBackend>(
            &runtime,
            &<TestBackend as burn::tensor::backend::Backend>::Device::default(),
        )
        .expect("resolve devices");

        assert_eq!(devices.len(), 2);
        assert!(devices.iter().all(|device| device == &devices[0]));
    }

    #[test]
    fn resolve_training_devices_uses_single_rank_local_device_for_process_group() {
        type TestBackend = NdArray<f32>;

        let runtime = super::ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            global_rank: 1,
            local_rank: 1,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };
        let devices = resolve_training_devices::<TestBackend>(
            &runtime,
            &<TestBackend as burn::tensor::backend::Backend>::Device::default(),
        )
        .expect("resolve devices");

        assert_eq!(devices.len(), 1);
        assert_eq!(
            devices[0],
            <TestBackend as burn::tensor::backend::Backend>::Device::default()
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_collective_config_includes_global_collective_settings() {
        let config = ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            data: crate::ParallelDataConfig {
                size: 2,
                collective_num_nodes: Some(2),
                collective_global_address: Some("127.0.0.1:32000".to_string()),
                collective_node_address: Some("127.0.0.1:32001".to_string()),
                collective_data_service_port: Some(32001),
                ..Default::default()
            },
            ..Default::default()
        };
        let runtime = resolve_parallel_runtime_with_env(&config, &ParallelEnv::default())
            .expect("local ddp bridge");

        let collective = resolve_collective_config(&runtime, &config).expect("collective config");
        let json = serde_json::to_value(&collective).expect("serialize collective config");

        assert_eq!(json["num_devices"], json!(2));
        assert_eq!(json["num_nodes"], json!(2));
        assert_eq!(
            json["global_address"]["inner"],
            json!("ws://127.0.0.1:32000")
        );
        assert_eq!(json["node_address"]["inner"], json!("ws://127.0.0.1:32001"));
        assert_eq!(json["data_service_port"], json!(32001));
        assert_eq!(json["global_all_reduce_strategy"], json!("Centralized"));
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_collective_config_preserves_explicit_ws_scheme() {
        let config = ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            data: crate::ParallelDataConfig {
                size: 2,
                collective_num_nodes: Some(2),
                collective_global_address: Some("ws://127.0.0.1:33000".to_string()),
                collective_node_address: Some("ws://127.0.0.1:33001".to_string()),
                collective_data_service_port: Some(33001),
                ..Default::default()
            },
            ..Default::default()
        };
        let runtime = resolve_parallel_runtime_with_env(&config, &ParallelEnv::default())
            .expect("local ddp bridge");

        let collective = resolve_collective_config(&runtime, &config).expect("collective config");
        let json = serde_json::to_value(&collective).expect("serialize collective config");

        assert_eq!(
            json["global_address"]["inner"],
            json!("ws://127.0.0.1:33000")
        );
        assert_eq!(json["node_address"]["inner"], json!("ws://127.0.0.1:33001"));
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_collective_config_requires_global_settings_for_process_group() {
        let config = ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: crate::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let env = ParallelEnv {
            world_size: Some(2),
            global_rank: Some(0),
            local_rank: Some(0),
        };
        let runtime =
            resolve_parallel_runtime_with_env(&config, &env).expect("process-group runtime");
        let err = resolve_collective_config(&runtime, &config)
            .expect_err("missing global collective settings should fail");
        assert!(
            err.to_string()
                .contains("require collective global settings"),
            "unexpected error: {err:#}"
        );
    }
}
