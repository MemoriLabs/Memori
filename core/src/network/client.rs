use std::env;
use std::time::Duration;

use reqwest::{Client, RequestBuilder};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;
use tokio::time::sleep;

use crate::network::error::ApiError;

const PUBLIC_PROD_KEY: &str = "96a7ea3e-11c2-428c-b9ae-5a168363dc80";
const PUBLIC_STAGING_KEY: &str = "c18b1022-7fe2-42af-ab01-b1f9139184f0";

pub enum ApiSubdomain {
    Default,
    Collector,
}

impl ApiSubdomain {
    fn as_str(&self) -> &'static str {
        match self {
            ApiSubdomain::Default => "api",
            ApiSubdomain::Collector => "collector",
        }
    }
}

#[derive(Clone)]
pub struct MemoriClient {
    client: Client,
    base_url: String,
    x_api_key: String,
    api_key: Option<String>,
}

impl MemoriClient {
    /// Initializes the client from environment variables.
    pub fn new(subdomain: ApiSubdomain) -> Result<Self, ApiError> {
        let prefix = subdomain.as_str();
        let is_staging = env::var("MEMORI_ENV").unwrap_or_default() == "staging";

        let (base_url, x_api_key) = {
            let x_key = env::var("MEMORI_X_API_KEY");

            if let Ok(domain) = env::var("MEMORI_DOMAIN") {
                let domain = domain.trim().to_string();
                if !domain.is_empty() {
                    let url = if is_staging {
                        format!("https://staging-{}.{}", prefix, domain)
                    } else {
                        format!("https://{}.{}", prefix, domain)
                    };
                    let key = x_key.unwrap_or_else(|_| {
                        if is_staging {
                            PUBLIC_STAGING_KEY
                        } else {
                            PUBLIC_PROD_KEY
                        }
                        .to_string()
                    });
                    (url, key)
                } else {
                    Self::resolve_fallback(prefix, is_staging, x_key)
                }
            } else {
                Self::resolve_fallback(prefix, is_staging, x_key)
            }
        };

        let api_key = env::var("MEMORI_API_KEY").ok();

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ApiError::Network(e.to_string()))?;

        Ok(Self {
            client,
            base_url,
            x_api_key,
            api_key,
        })
    }

    fn resolve_fallback(
        prefix: &str,
        is_staging: bool,
        x_key: Result<String, env::VarError>,
    ) -> (String, String) {
        if let Ok(url) = env::var("MEMORI_API_URL_BASE") {
            let url = url.trim().to_string();
            if !url.is_empty() {
                let key = x_key.unwrap_or_else(|_| PUBLIC_STAGING_KEY.to_string());
                return (url, key);
            }
        }
        let key = x_key.unwrap_or_else(|_| {
            if is_staging {
                PUBLIC_STAGING_KEY
            } else {
                PUBLIC_PROD_KEY
            }
            .to_string()
        });
        if is_staging {
            (format!("https://staging-{}.memorilabs.ai", prefix), key)
        } else {
            (format!("https://{}.memorilabs.ai", prefix), key)
        }
    }

    fn url(&self, route: &str) -> String {
        format!("{}/v1/{}", self.base_url, route)
    }

    fn build_request<T: Serialize>(&self, route: &str, payload: &T) -> RequestBuilder {
        let mut req = self
            .client
            .post(self.url(route))
            .header("X-Memori-API-Key", &self.x_api_key)
            .json(payload);

        if let Some(token) = &self.api_key {
            req = req.bearer_auth(token);
        }

        req
    }

    pub async fn post_async<T: Serialize, R: DeserializeOwned>(
        &self,
        route: &str,
        payload: &T,
    ) -> Result<R, ApiError> {
        let max_retries = 5;
        let mut attempts = 0;
        let backoff_factor = 1;

        loop {
            let req = self.build_request(route, payload);

            match req.send().await {
                Ok(response) => {
                    let status = response.status();

                    if status.is_server_error() {
                        if attempts >= max_retries {
                            log::error!("Max retries exceeded for {} error", status);
                            return Err(ApiError::Client {
                                status_code: status,
                                message: format!("Max retries exceeded for {} error", status),
                                details: None,
                            });
                        }
                        self.do_backoff(attempts, backoff_factor).await;
                        attempts += 1;
                        continue;
                    }

                    return self.handle_response(response).await;
                }
                Err(e) => {
                    if attempts >= max_retries {
                        // Check if the error is related to SSL/TLS certificates
                        let err_msg = e.to_string().to_lowercase();
                        if err_msg.contains("certificate")
                            || err_msg.contains("tls")
                            || err_msg.contains("ssl")
                            || err_msg.contains("handshake")
                        {
                            log::error!("SSL/TLS error during request: {}", e);
                            return Err(ApiError::Ssl(e.to_string()));
                        }

                        log::error!("Network error, max retries exceeded: {}", e);
                        return Err(ApiError::Network(e.to_string()));
                    }
                    self.do_backoff(attempts, backoff_factor).await;
                    attempts += 1;
                }
            }
        }
    }

    pub async fn post_async_raw<T: Serialize>(
        &self,
        route: &str,
        payload: &T,
    ) -> Result<String, ApiError> {
        let max_retries = 5;
        let mut attempts = 0;
        let backoff_factor = 1;

        loop {
            let req = self.build_request(route, payload);

            match req.send().await {
                Ok(response) => {
                    let status = response.status();

                    if status.is_server_error() {
                        if attempts >= max_retries {
                            log::error!("Max retries exceeded for {} error", status);
                            return Err(ApiError::Client {
                                status_code: status,
                                message: format!("Max retries exceeded for {} error", status),
                                details: None,
                            });
                        }
                        self.do_backoff(attempts, backoff_factor).await;
                        attempts += 1;
                        continue;
                    }

                    return self.handle_response_raw(response).await;
                }
                Err(e) => {
                    if attempts >= max_retries {
                        let err_msg = e.to_string().to_lowercase();
                        if err_msg.contains("certificate")
                            || err_msg.contains("tls")
                            || err_msg.contains("ssl")
                            || err_msg.contains("handshake")
                        {
                            log::error!("SSL/TLS error during request: {}", e);
                            return Err(ApiError::Ssl(e.to_string()));
                        }

                        log::error!("Network error, max retries exceeded: {}", e);
                        return Err(ApiError::Network(e.to_string()));
                    }
                    self.do_backoff(attempts, backoff_factor).await;
                    attempts += 1;
                }
            }
        }
    }

    pub async fn augmentation_async<T: Serialize, R: DeserializeOwned>(
        &self,
        payload: &T,
    ) -> Result<R, ApiError> {
        self.post_async("sdk/augmentation", payload).await
    }

    pub async fn augmentation_raw_async<T: Serialize>(
        &self,
        payload: &T,
    ) -> Result<String, ApiError> {
        self.post_async_raw("sdk/augmentation", payload).await
    }

    async fn do_backoff(&self, attempts: u32, factor: u64) {
        let sleep_secs = factor * (2_u64.pow(attempts));
        log::debug!(
            "Retrying after error in {}s (attempt {}/5)",
            sleep_secs,
            attempts + 1
        );
        sleep(Duration::from_secs(sleep_secs)).await;
    }

    async fn handle_response<R: DeserializeOwned>(
        &self,
        response: reqwest::Response,
    ) -> Result<R, ApiError> {
        let status = response.status();

        if status.is_success() {
            return response
                .json::<R>()
                .await
                .map_err(|e| ApiError::Network(e.to_string()));
        }

        let error_body: Option<Value> = response.json().await.ok();
        let message = error_body
            .as_ref()
            .and_then(|v| v.get("message").or_else(|| v.get("detail")))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("Memori API request failed with status {}", status));

        match status.as_u16() {
            429 => {
                let msg = if self.api_key.is_none() {
                    message
                } else {
                    "Quota exceeded".to_string()
                };
                Err(ApiError::QuotaExceeded(msg))
            }
            422 => Err(ApiError::Validation {
                message,
                details: error_body,
            }),
            433 => Err(ApiError::Rejected {
                message,
                details: error_body,
            }),
            _ => Err(ApiError::Client {
                status_code: status,
                message,
                details: error_body,
            }),
        }
    }

    async fn handle_response_raw(&self, response: reqwest::Response) -> Result<String, ApiError> {
        let status = response.status();
        let response_text = response
            .text()
            .await
            .map_err(|e| ApiError::Network(e.to_string()))?;

        if status.is_success() {
            return Ok(response_text);
        }

        let error_body: Option<Value> = serde_json::from_str::<Value>(&response_text).ok();
        let message = error_body
            .as_ref()
            .and_then(|v| v.get("message").or_else(|| v.get("detail")))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                if response_text.is_empty() {
                    format!("Memori API request failed with status {}", status)
                } else {
                    response_text.clone()
                }
            });

        match status.as_u16() {
            429 => {
                let msg = if self.api_key.is_none() {
                    message
                } else {
                    "Quota exceeded".to_string()
                };
                Err(ApiError::QuotaExceeded(msg))
            }
            422 => Err(ApiError::Validation {
                message,
                details: error_body,
            }),
            433 => Err(ApiError::Rejected {
                message,
                details: error_body,
            }),
            _ => Err(ApiError::Client {
                status_code: status,
                message,
                details: error_body,
            }),
        }
    }
}
