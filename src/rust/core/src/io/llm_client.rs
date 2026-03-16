use serde::{Deserialize, Serialize};
use reqwest::{Client, header};

#[derive(Serialize, Deserialize, Debug)]
pub struct StatePrompt {
    pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AgentAction {
    pub reasoning_trace: Vec<String>,
    pub intended_pyramid_layer: String,
    pub delta_torque_nm: f64,
    pub delta_flow_rate_lpm: f64,
    pub confidence: f64,
}

#[derive(Debug)]
pub enum LlmError {
    NetworkError(String),
    ParseError(String),
    ApiError(String),
}

pub struct LlmClient {
    base_url: String,
    api_key: String,
    model: String,
    client: Client,
}

impl LlmClient {
    pub fn new(base_url: String, api_key: String, model: String) -> Self {
        let client = Client::new();
        Self {
            base_url,
            api_key,
            model,
            client,
        }
    }

    pub async fn predict(&self, state: &StatePrompt) -> Result<AgentAction, LlmError> {
        // Builds a generic OpenAI-compatible completion request.
        // Most open-source (vLLM, Ollama) and proprietary endpoints support
        // the /chat/completions endpoint structure.
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a cyber-physical control agent executing within a strictly typed environment. You must output only valid JSON matching the schema: { \"reasoning_trace\": [string], \"intended_pyramid_layer\": string, \"delta_torque_nm\": float, \"delta_flow_rate_lpm\": float, \"confidence\": float }. Do not include any supplementary text."
                },
                {
                    "role": "user",
                    "content": state.prompt
                }
            ],
            // Request JSON mode formatting if the backend supports it natively
            "response_format": { "type": "json_object" },
            "temperature": 0.0
        });

        let mut request = self.client.post(&self.base_url)
            .header(header::CONTENT_TYPE, "application/json");
            
        if !self.api_key.is_empty() {
             request = request.header(header::AUTHORIZATION, format!("Bearer {}", self.api_key));
        }

        let res = request
            .json(&payload)
            .send()
            .await
            .map_err(|e| LlmError::NetworkError(e.to_string()))?;

        if !res.status().is_success() {
            let error_text = res.text().await.unwrap_or_default();
            return Err(LlmError::ApiError(error_text));
        }

        let body: serde_json::Value = res.json().await.map_err(|e| LlmError::ParseError(e.to_string()))?;
        
        // Extract the content using standard completion schema
        let content_str = body["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| LlmError::ParseError("Missing content string in response".to_string()))?;

        // Parse the strictly typed action struct
        let action: AgentAction = serde_json::from_str(content_str)
            .map_err(|e| LlmError::ParseError(format!("Invalid JSON schema: {}", e)))?;

        Ok(action)
    }
}
