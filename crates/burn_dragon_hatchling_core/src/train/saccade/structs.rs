use crate::train::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct LevelCoordsCacheState<B: BackendTrait> {
    pub(crate) map: HashMap<(usize, usize), Tensor<B, 2>>,
    pub(crate) order: VecDeque<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub(crate) struct LevelCoordsCache<B: BackendTrait> {
    pub(crate) inner: Arc<Mutex<LevelCoordsCacheState<B>>>,
    pub(crate) max_entries: usize,
}

impl<B: BackendTrait> LevelCoordsCache<B> {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(LevelCoordsCacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
            })),
            max_entries,
        }
    }

    pub(crate) fn get_or_build(&self, grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
        let key = (grid.height, grid.width);
        if let Ok(cache) = self.inner.lock() {
            if let Some(coords) = cache.map.get(&key) {
                return coords.clone();
            }
        }
        let coords = build_level_coords::<B>(grid, device);
        if self.max_entries == 0 {
            return coords;
        }
        if let Ok(mut cache) = self.inner.lock() {
            if !cache.map.contains_key(&key) {
                cache.order.push_back(key);
            }
            cache.map.insert(key, coords.clone());
            while cache.map.len() > self.max_entries {
                if let Some(evicted) = cache.order.pop_front() {
                    cache.map.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        coords
    }
}

impl<B: BackendTrait> Module<B> for LevelCoordsCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for LevelCoordsCache<B> {
    type InnerModule = LevelCoordsCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        LevelCoordsCache::new(self.max_entries)
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for LevelCoordsCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let max_entries = self.max_entries;
        let entries = self.inner.lock().map(|cache| cache.map.len()).unwrap_or(0);
        content
            .add("entries", &entries)
            .add("max_entries", &max_entries)
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for LevelCoordsCache<B> {}

#[derive(Clone, Debug)]
pub(crate) struct UpsampleWeightsCacheState<B: BackendTrait> {
    pub(crate) map: HashMap<(usize, usize, usize, usize), Tensor<B, 2>>,
    pub(crate) order: VecDeque<(usize, usize, usize, usize)>,
}

#[derive(Clone, Debug)]
pub(crate) struct UpsampleWeightsCache<B: BackendTrait> {
    pub(crate) inner: Arc<Mutex<UpsampleWeightsCacheState<B>>>,
    pub(crate) max_entries: usize,
}

impl<B: BackendTrait> UpsampleWeightsCache<B> {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(UpsampleWeightsCacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
            })),
            max_entries,
        }
    }

    pub(crate) fn get_or_build(
        &self,
        from: PatchGrid,
        to: PatchGrid,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let key = (from.height, from.width, to.height, to.width);
        if let Ok(cache) = self.inner.lock() {
            if let Some(weights) = cache.map.get(&key) {
                return weights.clone();
            }
        }
        let from_tokens = from.num_patches();
        let to_tokens = to.num_patches();
        let mut mapping = vec![0.0f32; to_tokens * from_tokens];
        for ty in 0..to.height {
            let src_y = (ty as f32 * from.height as f32 / to.height as f32)
                .floor()
                .min((from.height - 1) as f32) as usize;
            for tx in 0..to.width {
                let src_x = (tx as f32 * from.width as f32 / to.width as f32)
                    .floor()
                    .min((from.width - 1) as f32) as usize;
                let src_idx = src_y * from.width + src_x;
                let dst_idx = ty * to.width + tx;
                mapping[dst_idx * from_tokens + src_idx] = 1.0;
            }
        }
        let weights =
            Tensor::<B, 2>::from_data(TensorData::new(mapping, [to_tokens, from_tokens]), device);
        if self.max_entries == 0 {
            return weights;
        }
        if let Ok(mut cache) = self.inner.lock() {
            if !cache.map.contains_key(&key) {
                cache.order.push_back(key);
            }
            cache.map.insert(key, weights.clone());
            while cache.map.len() > self.max_entries {
                if let Some(evicted) = cache.order.pop_front() {
                    cache.map.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        weights
    }
}

impl<B: BackendTrait> Module<B> for UpsampleWeightsCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for UpsampleWeightsCache<B> {
    type InnerModule = UpsampleWeightsCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        UpsampleWeightsCache::new(self.max_entries)
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for UpsampleWeightsCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let max_entries = self.max_entries;
        let entries = self.inner.lock().map(|cache| cache.map.len()).unwrap_or(0);
        content
            .add("entries", &entries)
            .add("max_entries", &max_entries)
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for UpsampleWeightsCache<B> {}

#[derive(Clone, Debug)]
pub(crate) struct FoveaBaseGridCacheState<B: BackendTrait> {
    pub(crate) map: HashMap<usize, Tensor<B, 4>>,
    pub(crate) order: VecDeque<usize>,
}

#[derive(Clone, Debug)]
pub(crate) struct FoveaBaseGridCache<B: BackendTrait> {
    pub(crate) inner: Arc<Mutex<FoveaBaseGridCacheState<B>>>,
    pub(crate) max_entries: usize,
}

impl<B: BackendTrait> FoveaBaseGridCache<B> {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FoveaBaseGridCacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
            })),
            max_entries,
        }
    }

    pub(crate) fn get_or_build(&self, patch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        let key = patch_size.max(1);
        if let Ok(cache) = self.inner.lock() {
            if let Some(grid) = cache.map.get(&key) {
                return grid.clone();
            }
        }
        let grid = build_foveated_base_grid::<B>(key, device);
        if self.max_entries == 0 {
            return grid;
        }
        if let Ok(mut cache) = self.inner.lock() {
            if !cache.map.contains_key(&key) {
                cache.order.push_back(key);
            }
            cache.map.insert(key, grid.clone());
            while cache.map.len() > self.max_entries {
                if let Some(evicted) = cache.order.pop_front() {
                    cache.map.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        grid
    }
}

impl<B: BackendTrait> Module<B> for FoveaBaseGridCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for FoveaBaseGridCache<B> {
    type InnerModule = FoveaBaseGridCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        FoveaBaseGridCache::new(self.max_entries)
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for FoveaBaseGridCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let max_entries = self.max_entries;
        let entries = self.inner.lock().map(|cache| cache.map.len()).unwrap_or(0);
        content
            .add("entries", &entries)
            .add("max_entries", &max_entries)
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for FoveaBaseGridCache<B> {}

#[derive(Clone, Debug)]
pub(crate) struct FoveaJitter<B: BackendTrait> {
    pub(crate) batched: Tensor<B, 5>,
    pub(crate) sequential: Vec<Tensor<B, 4>>,
}

#[derive(Clone, Debug)]
pub(crate) struct FoveaJitterCacheState<B: BackendTrait> {
    pub(crate) map: HashMap<(usize, usize), FoveaJitter<B>>,
    pub(crate) order: VecDeque<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub(crate) struct FoveaJitterCache<B: BackendTrait> {
    pub(crate) inner: Arc<Mutex<FoveaJitterCacheState<B>>>,
    pub(crate) max_entries: usize,
}

impl<B: BackendTrait> FoveaJitterCache<B> {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(FoveaJitterCacheState {
                map: HashMap::new(),
                order: VecDeque::new(),
            })),
            max_entries,
        }
    }

    pub(crate) fn get_or_build(
        &self,
        patch_size: usize,
        subsamples_axis: usize,
        device: &B::Device,
    ) -> FoveaJitter<B> {
        let key = (patch_size.max(1), subsamples_axis.max(1));
        if let Ok(cache) = self.inner.lock() {
            if let Some(jitter) = cache.map.get(&key) {
                return jitter.clone();
            }
        }
        let jitter = build_fovea_jitter::<B>(key.0, key.1, device);
        if self.max_entries == 0 {
            return jitter;
        }
        if let Ok(mut cache) = self.inner.lock() {
            if !cache.map.contains_key(&key) {
                cache.order.push_back(key);
            }
            cache.map.insert(key, jitter.clone());
            while cache.map.len() > self.max_entries {
                if let Some(evicted) = cache.order.pop_front() {
                    cache.map.remove(&evicted);
                } else {
                    break;
                }
            }
        }
        jitter
    }
}

impl<B: BackendTrait> Module<B> for FoveaJitterCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for FoveaJitterCache<B> {
    type InnerModule = FoveaJitterCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        FoveaJitterCache::new(self.max_entries)
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for FoveaJitterCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let max_entries = self.max_entries;
        let entries = self.inner.lock().map(|cache| cache.map.len()).unwrap_or(0);
        content
            .add("entries", &entries)
            .add("max_entries", &max_entries)
            .optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for FoveaJitterCache<B> {}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) recon: VisionReconstructionHead<B>,
    // Learned initial trajectory state for the recurrent rollout.
    pub(crate) trajectory_token: Param<Tensor<B, 2>>,
    // Per-eye identity bias to keep multi-eye rollouts disentangled.
    pub(crate) eye_token: Param<Tensor<B, 2>>,
    pub(crate) input_proj: VisionSaccadeInputProjection<B>,
    pub(crate) fovea_proj: VisionSaccadeProjection<B>,
    pub(crate) pyramid_in_proj: Option<VisionSaccadeProjection<B>>,
    pub(crate) pyramid_out_proj: Option<VisionSaccadeProjection<B>>,
    pub(crate) residual_proj: VisionSaccadeProjection<B>,
    pub(crate) saccade_head: VisionSaccadeHead<B>,
    pub(crate) config: VisionSaccadeConfig,
    pub(crate) level_coords_cache: LevelCoordsCache<B>,
    pub(crate) upsample_weights_cache: UpsampleWeightsCache<B>,
    pub(crate) fovea_grid_cache: FoveaBaseGridCache<B>,
    pub(crate) fovea_jitter_cache: FoveaJitterCache<B>,
    #[module(ignore)]
    pub(crate) pyramid_dim: usize,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
    #[module(ignore)]
    pub(crate) train_repeats: usize,
    #[module(ignore)]
    pub(crate) train_repeat_chunk: usize,
}

pub(crate) struct VisionSaccadeLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) inv: Tensor<B, 1>,
    pub(crate) sigreg: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

pub(crate) struct GdpoPolicyInputs<B: BackendTrait> {
    pub(crate) hard_reward: Tensor<B, 1>,
    pub(crate) recon_per_sample: Tensor<B, 1>,
    pub(crate) log_prob_sum: Tensor<B, 2>,
    pub(crate) log_prob_sum_old: Tensor<B, 2>,
    pub(crate) gdpo_group: usize,
}

pub(crate) struct SaccadeMipLevel<B: BackendTrait> {
    pub(crate) tokens: Tensor<B, 3>,
    pub(crate) grid: PatchGrid,
    pub(crate) image: Tensor<B, 4>,
}

pub(crate) struct SaccadeLaplacianImages<B: BackendTrait> {
    pub(crate) residuals: Vec<Tensor<B, 4>>,
    pub(crate) coarse: Tensor<B, 4>,
}


