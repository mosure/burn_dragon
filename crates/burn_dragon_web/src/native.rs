type JsValue = String;

fn unsupported() -> JsValue {
    "burn_dragon_web is only available on wasm32 targets".to_string()
}

pub struct WasmInference;

pub async fn load_model(
    _model_bytes: Vec<u8>,
    _vocab_json: String,
    _config_json: Option<String>,
    _start_viz: Option<bool>,
) -> Result<WasmInference, JsValue> {
    Err(unsupported())
}

impl WasmInference {
    pub fn start_stream(
        &mut self,
        _prompt: String,
        _max_tokens: Option<i32>,
        _temperature: f32,
        _top_k: Option<u32>,
        _context_window: Option<u32>,
    ) -> Result<String, JsValue> {
        Err(unsupported())
    }

    pub async fn next_chunk(&mut self) -> Result<Option<String>, JsValue> {
        Err(unsupported())
    }

    pub fn clear_stream(&mut self) {}
}
