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

/// Resolves the API base URL from environment variables.
///
/// Priority:
/// 1. `MEMORI_ENTERPRISE_PRODUCTION_DOMAIN` → `https://{subdomain}.{domain}`
/// 2. `MEMORI_ENTERPRISE_STAGING_DOMAIN`    → `https://staging-{subdomain}.{domain}`
/// 3. `MEMORI_API_URL_BASE`                 → verbatim (backward compat)
/// 4. `MEMORI_TEST_MODE=1`                  → `https://staging-{subdomain}.memorilabs.ai`
/// 5. Default                               → `https://{subdomain}.memorilabs.ai`
pub fn resolve_base_url(subdomain: &ApiSubdomain) -> String {
    let prefix = subdomain.as_str();

    if let Ok(domain) = env::var("MEMORI_ENTERPRISE_PRODUCTION_DOMAIN") {
        let domain = domain.trim().to_string();
        if !domain.is_empty() {
            return format!("https://{}.{}", prefix, domain);
        }
    }

    if let Ok(domain) = env::var("MEMORI_ENTERPRISE_STAGING_DOMAIN") {
        let domain = domain.trim().to_string();
        if !domain.is_empty() {
            return format!("https://staging-{}.{}", prefix, domain);
        }
    }

    if let Ok(url) = env::var("MEMORI_API_URL_BASE") {
        let url = url.trim().to_string();
        if !url.is_empty() {
            return url;
        }
    }

    let test_mode = env::var("MEMORI_TEST_MODE").unwrap_or_default() == "1";
    if test_mode {
        format!("https://staging-{}.memorilabs.ai", prefix)
    } else {
        format!("https://{}.memorilabs.ai", prefix)
    }
}

/// Resolves the X-Memori-API-Key from environment variables.
///
/// Returns the production key for enterprise production and default production,
/// the staging key for all other cases. Always respects `MEMORI_X_API_KEY` override.
pub fn resolve_x_api_key() -> String {
    let override_key = env::var("MEMORI_X_API_KEY");

    let enterprise_prod = env::var("MEMORI_ENTERPRISE_PRODUCTION_DOMAIN")
        .map(|d| !d.trim().is_empty())
        .unwrap_or(false);

    if enterprise_prod {
        return override_key.unwrap_or_else(|_| PUBLIC_PROD_KEY.to_string());
    }

    let uses_staging = env::var("MEMORI_ENTERPRISE_STAGING_DOMAIN")
        .map(|d| !d.trim().is_empty())
        .unwrap_or(false)
        || env::var("MEMORI_API_URL_BASE")
            .map(|u| !u.trim().is_empty())
            .unwrap_or(false)
        || env::var("MEMORI_TEST_MODE").unwrap_or_default() == "1";

    if uses_staging {
        override_key.unwrap_or_else(|_| PUBLIC_STAGING_KEY.to_string())
    } else {
        override_key.unwrap_or_else(|_| PUBLIC_PROD_KEY.to_string())
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
        let base_url = resolve_base_url(&subdomain);
        let x_api_key = resolve_x_api_key();
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
